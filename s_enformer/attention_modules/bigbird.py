from typing import List, Optional

import numpy as np
import sonnet as snt
import tensorflow as tf

from s_enformer.utils import modelling as utils


MAX_SEQ_LEN = 4096


class MultiheadAttention(snt.Module):
    """Multi-head attention."""

    def __init__(self,
                 channels: int,
                 value_size: int,
                 key_size: int,
                 num_heads: int,
                 scaling: bool = True,
                 attention_dropout_rate: float = 0.1,
                 relative_positions: bool = False,
                 relative_position_symmetric: bool = False,
                 relative_position_functions: Optional[List[str]] = None,
                 num_relative_position_features: Optional[int] = None,
                 positional_dropout_rate: float = 0.1,

                 use_bias: bool = True,
                 initializer_range=0.02,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 seed=None,

                 zero_initialize: bool = True,
                 initializer: Optional[snt.initializers.Initializer] = None,
                 name: str = None):
        """Creates a MultiheadAttention module.

        Args:
            value_size: The size of each value embedding per head.
            key_size: The size of each key and query embedding per head.
            num_heads: The number of independent queries per timestep.
            scaling: Whether to scale the attention logits.
            attention_dropout_rate: Dropout rate for attention logits.
            relative_positions: Whether to use TransformerXL style relative attention.
            relative_position_symmetric: If True, the symmetric version of basis
                functions will be used. If False, a symmetric and asymmetric versions
                will be use.
            relative_position_functions: List of function names used for relative
                positional biases.
            num_relative_position_features: Number of relative positional features
                to compute. If None, `value_size * num_heads` is used.
            positional_dropout_rate: Dropout rate for the positional encodings if
                relative positions are used.
            zero_initialize: if True, the final linear layer will be 0 initialized.
            initializer: Initializer for the projection layers. If unspecified,
                VarianceScaling is used with scale = 2.0.
            name: Name of module.
        """

        super().__init__(name=name)

        # Need to change seq_length if the input sequence length to the model changes
        seq_length = 1536
        self.from_seq_length = seq_length
        self.to_seq_length = seq_length
        self.from_block_size = 64
        self.to_block_size = 64
        self.num_rand_blocks = 3  # Default value: https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py#L650
        self.num_attention_heads = num_heads
        self.size_per_head = channels // self.num_attention_heads
        self.seed = seed

        hidden_size = channels
        attention_head_size = hidden_size // self.num_attention_heads

        self._q_layer = utils.Dense3dLayer(
            self.num_attention_heads, self.size_per_head,
            utils.create_initializer(initializer_range), query_act,
            "q_layer", head_first=True, use_bias=use_bias)

        self._k_layer = utils.Dense3dLayer(
            self.num_attention_heads, self.size_per_head,
            utils.create_initializer(initializer_range), key_act,
            "k_layer", head_first=True, use_bias=use_bias)

        self._v_layer = utils.Dense3dLayer(
            self.num_attention_heads, self.size_per_head,
            utils.create_initializer(initializer_range), value_act,
            "v_layer", head_first=True, use_bias=use_bias)

        self.projection_layer = utils.Dense3dProjLayer(
            self.num_attention_heads, attention_head_size,
            utils.create_initializer(initializer_range), None,
            "dense", use_bias)

        self.rand_attn = self.generate_rand_attn_list()
        self.attn_impl = self.bigbird_block_sparse_attention

    def bigbird_block_sparse_attention(self,
                                       query_layer,
                                       key_layer,
                                       value_layer,
                                       masks,
                                       training=None):
        """BigBird attention sparse calculation using blocks in linear time.

        Args:
        query_layer: float Tensor of shape [batch_size, num_attention_heads,
            from_seq_length, size_per_head]
        key_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
        value_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
        masks: A list of 5 masks used in BigBird attention at position 1 to 5.
          Position 0 (first element) is not used can be left as none. In the mask,
          the values should be 1 or 0. The attention scores will effectively
          be set to -infinity for any positions in the mask that are 0,
          and will be unchanged for positions that are 1.
             "None": Not needed.
              "band_mask": (optional) float32 Tensor of shape
                  [batch_size, 1, from_seq_length//from_block_size-4,
                  from_block_size, 3*to_block_size].
              "from_mask": (optional) float32 Tensor of shape
                  [batch_size, 1, from_seq_length, 1].
              "to_mask": (optional) float32 Tensor of shape
                  [batch_size, 1, 1, to_seq_length].
              "from_blocked_mask": (optional) float32 Tensor of shape
                  [batch_size, from_seq_length//from_block_size, from_block_size].
                  Same as from_mask, just reshaped.
              "to_blocked_mask": (optional) float32 Tensor of shape
                  [batch_size, to_seq_length//to_block_size, to_block_size].
                  Same as to_mask, just reshaped.}
        training: Boolean indicating whether the call is training or inference.

        Returns:
        float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
            size_per_head].
        """
        (band_mask, from_mask, to_mask,
         from_blocked_mask, to_blocked_mask) = masks

        attention_output = bigbird_block_sparse_attention(
            query_layer, key_layer, value_layer,
            band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
            self.rand_attn, self.num_attention_heads, self.size_per_head,
            self.num_rand_blocks, self.from_seq_length, self.to_seq_length,
            self.from_block_size, self.to_block_size)

        attention_output = self.projection_layer(attention_output)

        return attention_output

    def __call__(self,
                 from_tensor,
                 cache=None,
                 decode_i=None,
                 training=None):
        """Implements a multi-headed attention layer from from_tensor to to_tensor.
        Args:
            from_tensor: float Tensor of shape [batch_size, from_seq_length,
                from_width]
            cache: (Used during prediction) A dictionary with tensors containing
                results of previous attentions. The dictionary must have the items:
                    {"k": tensor with shape
                        [batch_size, max_len, num_attention_heads, size_per_head],
                    "v": tensor with shape
                        [batch_size, max_len, num_attention_heads, size_per_head]}
            decode_i: (Used during prediction) current location of decoding
            training: Boolean indicating whether the call is training or inference.
        Returns:
            float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
                size_per_head].
        Raises:
            ValueError: Any of the arguments or tensor shapes are invalid.
            NotImplementedError: For unknown attention type.
        """

        batch_size = utils.get_shape_list(from_tensor)[0]
        input_length = utils.get_shape_list(from_tensor)[1]

        encoder_inputs_mask = tf.ones([batch_size, input_length])

        encoder_length = input_length
        encoder_block_size = self.from_block_size
        encoder_inputs_mask = tf.cast(encoder_inputs_mask, tf.float32)

        blocked_encoder_mask = tf.reshape(
            encoder_inputs_mask,
            (batch_size, encoder_length // encoder_block_size, encoder_block_size))

        encoder_from_mask = tf.reshape(
            encoder_inputs_mask, (batch_size, 1, encoder_length, 1)
        )

        encoder_to_mask = tf.reshape(
            encoder_inputs_mask, (batch_size, 1, 1, encoder_length)
        )

        band_mask = create_band_mask_from_inputs(
            blocked_encoder_mask, blocked_encoder_mask
        )

        masks = (band_mask, encoder_from_mask, encoder_to_mask, blocked_encoder_mask, blocked_encoder_mask)

        # Scalar dimensions referenced here:
        #   b = batch size (number of sequences)
        #   m = `from_tensor` sequence length
        #   n = `to_tensor` sequence length
        #   h = `num_attention_heads`
        #   d = `size_per_head`

        # `query` = [b, h, m, d]
        query = self._q_layer(from_tensor)

        # `key` = [b, h, n, d]
        key = self._k_layer(from_tensor)

        # `value_layer` = [b, h, n, d]
        value = self._v_layer(from_tensor)

        if cache is not None and decode_i is not None:

            max_len = utils.get_shape_list(cache["k"])[2]

            indices_select = tf.reshape(
                tf.one_hot(decode_i, max_len, dtype=from_tensor.dtype),
                [1, 1, max_len, 1])

            key = cache["k"] + key * indices_select
            value = cache["v"] + value * indices_select
            cache["k"] = key
            cache["v"] = value

        contextual_output = self.attn_impl(
            query, key, value, masks, training=training
        )

        return contextual_output

    def generate_rand_attn_list(self):
        # generate random attention and corresponding masks
        if self.seed is not None:
            np.random.seed(self.seed)
        # old plans used in paper
        if self.from_seq_length in [1024, 2048, 3072, 4096]:
            rand_attn = [
                bigbird_block_rand_mask(  # pylint: disable=g-complex-comprehension
                    MAX_SEQ_LEN, MAX_SEQ_LEN,
                    self.from_block_size, self.to_block_size, self.num_rand_blocks,
                    last_idx=1024
                )[:(self.from_seq_length // self.from_block_size - 2)]
                for _ in range(self.num_attention_heads)
            ]
        else:
            plan_from_length, plan_num_rand_blocks = get_rand_attn_plan(
                self.from_seq_length, self.from_block_size, self.num_rand_blocks
            )
            rand_attn = bigbird_block_rand_mask_with_head(
                seq_length=self.from_seq_length,
                block_size=self.from_block_size,
                num_heads=self.num_attention_heads,
                plan_from_length=plan_from_length,
                plan_num_rand_blocks=plan_num_rand_blocks
            )
        rand_attn = np.stack(rand_attn, axis=0)

        return tf.constant(rand_attn, dtype=tf.int32)


def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
    """Create adjacency list of random attention.

    Args:
        from_seq_length: int. length of from sequence.
        to_seq_length: int. length of to sequence.
        from_block_size: int. size of block in from sequence.
        to_block_size: int. size of block in to sequence.
        num_rand_blocks: int. Number of random chunks per row.
        last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks choosen only upto last_idx.

    Returns:
        adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
      """
    rand_attn = np.zeros(
        (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32
    )
    middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = to_seq_length // to_block_size - 1
    if last_idx > (2 * to_block_size):
        last = (last_idx // to_block_size) - 1

    r = num_rand_blocks  # shorthand
    for i in range(1, from_seq_length // from_block_size - 1):
        start = i - 2
        end = i
        if i == 1:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_seq_length // from_block_size - 3:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
        elif i == from_seq_length // from_block_size - 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            elif (end + 1) == last:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            else:
                rand_attn[i - 1, :] = np.random.permutation(
                    np.concatenate((middle_seq[:start], middle_seq[end + 1:last]))
                )[:r]
    return rand_attn


def get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
    """Gives the plan of where to put random attention.

    Args:
        from_seq_length: int. length of from sequence.
        from_block_size: int. size of block in from sequence.
        num_rand_blocks: int. Number of random chunks per row.

    Returns:
        plan_from_length: ending location of from block
        plan_num_rand_blocks: number of random ending location for each block
    """
    # general plan
    plan_from_length = []
    plan_num_rand_blocks = []
    if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(0)
    elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks // 2)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
    else:
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks)

    return plan_from_length, plan_num_rand_blocks


def bigbird_block_rand_mask_with_head(seq_length,
                                      block_size,
                                      num_heads,
                                      plan_from_length,
                                      plan_num_rand_blocks,
                                      window_block_left=1,
                                      window_block_right=1,
                                      global_block_top=1,
                                      global_block_bottom=1,
                                      global_block_left=1,
                                      global_block_right=1):
    """Create adjacency list of random attention.

    Args:
        seq_length: int. length of sequence.
        block_size: int. size of block in sequence.
        num_heads: int. total number of heads.
        plan_from_length: list. plan from lenght where num_rand are choosen from.
        plan_num_rand_blocks: list. number of rand blocks within the plan.
        window_block_left: int. number of blocks of window to left of a block.
        window_block_right: int. number of blocks of window to right of a block.
        global_block_top: int. number of blocks at the top.
        global_block_bottom: int. number of blocks at the bottom.
        global_block_left: int. Number of blocks globally used to the left.
        global_block_right: int. Number of blocks globally used to the right.

    Returns:
        adjacency list of size num_head where each element is of size
        from_seq_length//from_block_size-2 by num_rand_blocks
    """
    # Total number of blocks in the mmask
    num_blocks = seq_length // block_size
    # Number of blocks per plan
    plan_block_length = np.array(plan_from_length) // block_size
    # till when to follow plan
    max_plan_idx = plan_from_length.index(seq_length)
    # Random Attention adjajency list
    rand_attn = [np.zeros((
        num_blocks,
        np.sum(plan_num_rand_blocks[:max_plan_idx + 1])
    ), dtype=np.int32) for i in range(num_heads)]

    # We will go iteratively over the plan blocks and pick random number of
    # Attention blocks from the legally allowed blocks
    for plan_idx in range(max_plan_idx + 1):
        rnd_r_cnt = 0
        if plan_idx > 0:
            # set the row for all from_blocks starting from 0 to
            # plan_block_length[plan_idx-1]
            # column indx start fromm plan_block_length[plan_idx-1] and ends at
            # plan_block_length[plan_idx]
            if plan_num_rand_blocks[plan_idx] > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
                for blk_rw_idx in range(
                    global_block_top, plan_block_length[plan_idx - 1]
                ):
                    for h in range(num_heads):

                        rand_attn[h][blk_rw_idx,
                                     rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                                         block_id=blk_rw_idx,
                                         to_start_block_id=plan_block_length[plan_idx - 1],
                                         to_end_block_id=plan_block_length[plan_idx],
                                         num_rand_blocks=plan_num_rand_blocks[plan_idx],
                                         window_block_left=window_block_left,
                                         window_block_right=window_block_right,
                                         global_block_left=global_block_left,
                                         global_block_right=global_block_right
                        )

            for pl_id in range(plan_idx):
                if plan_num_rand_blocks[pl_id] == 0:
                    continue
                for blk_rw_idx in range(plan_block_length[plan_idx - 1],
                                        plan_block_length[plan_idx]):
                    rnd_r_cnt = 0
                    to_start_block_id = 0
                    if pl_id > 0:
                        rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                        to_start_block_id = plan_block_length[pl_id - 1]
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id + 1]))
                    for h in range(num_heads):

                        rand_attn[h][blk_rw_idx,
                                     rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                                         block_id=blk_rw_idx,
                                         to_start_block_id=to_start_block_id,
                                         to_end_block_id=plan_block_length[pl_id],
                                         num_rand_blocks=plan_num_rand_blocks[pl_id],
                                         window_block_left=window_block_left,
                                         window_block_right=window_block_right,
                                         global_block_left=global_block_left,
                                         global_block_right=global_block_right
                        )

        if plan_num_rand_blocks[plan_idx] == 0:
            continue

        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
        from_start_block_id = global_block_top
        to_start_block_id = 0
        if plan_idx > 0:
            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
            from_start_block_id = plan_block_length[plan_idx - 1]
            to_start_block_id = plan_block_length[plan_idx - 1]

        for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
            for h in range(num_heads):

                rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = (
                    get_single_block_row_attention(
                        block_id=blk_rw_idx,
                        to_start_block_id=to_start_block_id,
                        to_end_block_id=plan_block_length[plan_idx],
                        num_rand_blocks=plan_num_rand_blocks[plan_idx],
                        window_block_left=window_block_left,
                        window_block_right=window_block_right,
                        global_block_left=global_block_left,
                        global_block_right=global_block_right
                    )
                )

    for nh in range(num_heads):
        rand_attn[nh] = rand_attn[nh][
            global_block_top:num_blocks - global_block_bottom, :
        ]
    return rand_attn


def get_single_block_row_attention(block_id,
                                   to_start_block_id,
                                   to_end_block_id,
                                   num_rand_blocks,
                                   window_block_left=1,
                                   window_block_right=1,
                                   global_block_left=1,
                                   global_block_right=1):
    """For a single row block get random row attention.

    Args:
        block_id: int. block id of row.
        to_start_block_id: int. random attention coloum start id.
        to_end_block_id: int. random attention coloum end id.
        num_rand_blocks: int. number of random blocks to be selected.
        window_block_left: int. number of blocks of window to left of a block.
        window_block_right: int. number of blocks of window to right of a block.
        global_block_left: int. Number of blocks globally used to the left.
        global_block_right: int. Number of blocks globally used to the right.

    Returns:
        row containing the random attention vector of size num_rand_blocks.
    """

    # list of to_blocks from which to choose random attention
    to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
    # permute the blocks
    perm_block = np.random.permutation(to_block_list)

    # illegal blocks for the current block id, using window
    illegal_blocks = list(
        range(block_id - window_block_left, block_id + window_block_right + 1))

    # Add blocks at the start and at the end
    illegal_blocks.extend(list(range(global_block_left)))
    illegal_blocks.extend(
        list(range(to_end_block_id - global_block_right, to_end_block_id)))

    # The second from_block cannot choose random attention on second last to_block
    if block_id == 1:
        illegal_blocks.append(to_end_block_id - 2)

    # The second last from_block cannot choose random attention on second to_block
    if block_id == to_end_block_id - 2:
        illegal_blocks.append(1)

    selected_random_blokcs = []

    for i in range(to_end_block_id - to_start_block_id):
        if perm_block[i] not in illegal_blocks:
            selected_random_blokcs.append(perm_block[i])
        if len(selected_random_blokcs) == num_rand_blocks:
            break
    return np.array(selected_random_blokcs, dtype=np.int32)


def bigbird_block_sparse_attention(query_layer,
                                   key_layer,
                                   value_layer,
                                   band_mask,
                                   from_mask,
                                   to_mask,
                                   from_blocked_mask,
                                   to_blocked_mask,
                                   rand_attn,
                                   num_attention_heads,
                                   size_per_head,
                                   num_rand_blocks,
                                   from_seq_length,
                                   to_seq_length,
                                   from_block_size,
                                   to_block_size):
    """BigBird attention sparse calculation using blocks in linear time.
    Assumes from_seq_length//from_block_size == to_seq_length//to_block_size.
    A pure function with a long argument list to allow easy use outside our
    framework.
    Args:
        query_layer: float Tensor of shape [batch_size, num_attention_heads,
            from_seq_length, size_per_head]
        key_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
        value_layer: float Tensor of shape [batch_size, num_attention_heads,
            to_seq_length, size_per_head]
        band_mask: float32 Tensor of shape [batch_size, 1,
            from_seq_length//from_block_size-4, from_block_size, 3*to_block_size].
            The values should be 1 or 0. The attention scores will effectively be
            set to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
        from_mask: float32 Tensor of shape [batch_size, 1, from_seq_length, 1].
            The values should be 1 or 0. The attention scores will effectively be set
            to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
        to_mask: float32 Tensor of shape [batch_size, 1, 1, to_seq_length].
            The values should be 1 or 0. The attention scores will effectively be set
            to -infinity for any positions in the mask that are 0, and will be
            unchanged for positions that are 1.
        from_blocked_mask: float32 Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            Same as from_mask, just reshaped.
        to_blocked_mask: float32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].
            Same as to_mask, just reshaped.
        rand_attn: int32 Tensor of shape [num_attention_heads,
            from_seq_length//from_block_size-2, num_rand_blocks] specifying which
            blocks to attend to for each from sequence block (except 2 global ones).
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        num_rand_blocks: int. Number of random chunks per row.
        from_seq_length: int. length of from sequence.
        to_seq_length: int. length of to sequence.
        from_block_size: int. size of block in from sequence.
        to_block_size: int. size of block in to sequence.
    Returns:
        float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].
    """
    assert from_seq_length // from_block_size == to_seq_length // to_block_size

    # repeat for batch size
    batch_size = utils.get_shape_list(query_layer)[0]

    rand_attn = tf.expand_dims(rand_attn, 0)
    rand_attn = tf.repeat(rand_attn, batch_size, 0)

    rand_mask = create_rand_mask_from_inputs(
        from_blocked_mask, to_blocked_mask, rand_attn,
        num_attention_heads, num_rand_blocks,
        from_seq_length, from_block_size)

    # Define shorthands
    # b = batch_size
    h = num_attention_heads
    r = num_rand_blocks
    d = size_per_head
    m = from_seq_length
    n = to_seq_length
    wm = from_block_size
    wn = to_block_size

    blocked_query_matrix = tf.reshape(query_layer, (-1, h, m // wm, wm, d))
    blocked_key_matrix = tf.reshape(key_layer, (-1, h, n // wn, wn, d))
    blocked_value_matrix = tf.reshape(value_layer, (-1, h, n // wn, wn, d))

    gathered_key = tf.reshape(
        tf.gather(blocked_key_matrix, rand_attn, batch_dims=2, name="gather_key"),
        (-1, h, m // wm - 2, r * wn, d))  # [b, h, n//wn-2, r, wn, -1]

    gathered_value = tf.reshape(
        tf.gather(
            blocked_value_matrix, rand_attn, batch_dims=2, name="gather_value"),
        (-1, h, m // wm - 2, r * wn, d))  # [b, h, n//wn-2, r, wn, -1]

    first_product = tf.einsum(
        "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 0],
        key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
    first_product = tf.multiply(first_product, 1.0 / np.sqrt(d))
    first_product += (1.0 - to_mask) * -10000.0
    first_attn_weights = tf.nn.softmax(first_product)  # [b, h, wm, n]
    first_context_layer = tf.einsum(
        "BHQK,BHKD->BHQD", first_attn_weights,
        value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
    first_context_layer = tf.expand_dims(first_context_layer, 2)

    second_key_mat = tf.concat([
        blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1],
        blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :, -1],
        gathered_key[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1]
    second_value_mat = tf.concat([
        blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1],
        blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1],
        gathered_value[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1]
    second_product = tf.einsum(
        "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 1], second_key_mat
    )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
    second_seq_pad = tf.concat([
        to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:],
        tf.ones_like(rand_mask[:, :1, 0, :1])], 3)
    second_rand_pad = tf.concat(
        [tf.ones_like(second_product[:, :, :, :4 * wn]), rand_mask[:, :, 0]], 3)
    second_product = tf.multiply(second_product, 1.0 / np.sqrt(d))
    second_product += (1.0 - tf.minimum(second_seq_pad, second_rand_pad)) * -10000.0
    second_attn_weights = tf.nn.softmax(second_product)  # [b , h, wm, (4+r)*wn]
    second_context_layer = tf.einsum(
        "BHQK,BHKD->BHQD", second_attn_weights, second_value_mat
    )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
    second_context_layer = tf.expand_dims(second_context_layer, 2)

    exp_blocked_key_matrix = tf.concat([
        blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2],
        blocked_key_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1]
    exp_blocked_value_matrix = tf.concat([
        blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2],
        blocked_value_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1]
    middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
    inner_band_product = tf.einsum(
        "BHLQD,BHLKD->BHLQK", middle_query_matrix, exp_blocked_key_matrix
    )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, 3*wn, -1]
    #     ==> [b, h, m//wm-4, wm, 3*wn]
    inner_band_product = tf.multiply(inner_band_product, 1.0 / np.sqrt(d))
    rand_band_product = tf.einsum(
        "BHLQD,BHLKD->BHLQK", middle_query_matrix, gathered_key[:, :, 1:-1]
    )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, r*wn, -1]
    #     ==> [b, h, m//wm-4, wm, r*wn]
    rand_band_product = tf.multiply(rand_band_product, 1.0 / np.sqrt(d))
    first_band_product = tf.einsum(
        "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, 0]
    )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
    first_band_product = tf.multiply(first_band_product, 1.0 / np.sqrt(d))
    last_band_product = tf.einsum(
        "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, -1]
    )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
    last_band_product = tf.multiply(last_band_product, 1.0 / np.sqrt(d))
    inner_band_product += (1.0 - band_mask) * -10000.0
    first_band_product += (
        1.0 - tf.expand_dims(to_mask[:, :, :, :wn], 3)) * -10000.0
    last_band_product += (
        1.0 - tf.expand_dims(to_mask[:, :, :, -wn:], 3)) * -10000.0
    rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0
    band_product = tf.concat([
        first_band_product, inner_band_product, rand_band_product,
        last_band_product], -1)  # [b, h, m//wm-4, wm, (5+r)*wn]
    attn_weights = tf.nn.softmax(band_product)  # [b, h, m//wm-4, wm, (5+r)*wn]
    context_layer = tf.einsum(
        "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :, wn:4 * wn],
        exp_blocked_value_matrix
    )  # [b, h, m//wm-4, wm, 3*wn] x [b, h, m//wm-4, 3*wn, -1]
    #     ==> [b, h, m//wm-4, wm, -1]
    context_layer += tf.einsum(
        "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :, 4 * wn:-wn],
        gathered_value[:, :, 1:-1]
    )  # [b, h, m//wm-4, wm, r*wn] x [b, h, m//wm-4, r*wn, -1]
    #     ==> [b, h, m//wm-4, wm, -1]
    context_layer += tf.einsum(
        "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :, :wn],
        blocked_value_matrix[:, :, 0]
    )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]
    context_layer += tf.einsum(
        "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :, -wn:],
        blocked_value_matrix[:, :, -1]
    )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]

    second_last_key_mat = tf.concat([
        blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3],
        blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1],
        gathered_key[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1]
    second_last_value_mat = tf.concat([
        blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3],
        blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1],
        gathered_value[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1]
    second_last_product = tf.einsum(
        "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -2], second_last_key_mat
    )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
    second_last_seq_pad = tf.concat([
        to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
        tf.ones_like(rand_mask[:, :1, 0, :1])], 3)
    second_last_rand_pad = tf.concat(
        [tf.ones_like(second_last_product[:, :, :, :4 * wn]), rand_mask[:, :, -1]], 3)
    second_last_product = tf.multiply(second_last_product, 1.0 / np.sqrt(d))
    second_last_product += (
        1.0 - tf.minimum(second_last_seq_pad, second_last_rand_pad)) * -10000.0
    second_last_attn_weights = tf.nn.softmax(
        second_last_product)  # [b, h, wm, (4+r)*wn]
    second_last_context_layer = tf.einsum(
        "BHQK,BHKD->BHQD", second_last_attn_weights, second_last_value_mat
    )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
    second_last_context_layer = tf.expand_dims(second_last_context_layer, 2)

    last_product = tf.einsum(
        "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -1],
        key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
    last_product = tf.multiply(last_product, 1.0 / np.sqrt(d))
    last_product += (1.0 - to_mask) * -10000.0
    last_attn_weights = tf.nn.softmax(last_product)  # [b, h, wm, n]
    last_context_layer = tf.einsum(
        "BHQK,BHKD->BHQD", last_attn_weights,
        value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
    last_context_layer = tf.expand_dims(last_context_layer, 2)

    context_layer = tf.concat([
        first_context_layer, second_context_layer, context_layer,
        second_last_context_layer, last_context_layer
    ], 2)
    context_layer = tf.reshape(context_layer, (-1, h, m, d)) * from_mask
    context_layer = tf.transpose(context_layer, (0, 2, 1, 3))

    return context_layer


def create_rand_mask_from_inputs(from_blocked_mask,
                                 to_blocked_mask,
                                 rand_attn,
                                 num_attention_heads,
                                 num_rand_blocks,
                                 from_seq_length,
                                 from_block_size):
    """Create 4D attention mask from a 3D tensor mask.
    Args:
        from_blocked_mask: 2D Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
        to_blocked_mask: int32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].
        rand_attn: [batch_size, num_attention_heads,
            from_seq_length//from_block_size-2, num_rand_blocks]
        num_attention_heads: int. Number of attention heads.
        num_rand_blocks: int. Number of random chunks per row.
        from_seq_length: int. length of from sequence.
        from_block_size: int. size of block in from sequence.
    Returns:
        float Tensor of shape [batch_size, num_attention_heads,
                               from_seq_length//from_block_size-2,
                               from_block_size, num_rand_blocks*to_block_size].
    """

    num_windows = from_seq_length // from_block_size - 2
    rand_mask = tf.reshape(
        tf.gather(to_blocked_mask, rand_attn, batch_dims=1), [
            -1, num_attention_heads, num_windows,
            num_rand_blocks * from_block_size
        ])

    rand_mask = tf.einsum("BLQ,BHLK->BHLQK", from_blocked_mask[:, 1:-1],
                          rand_mask)
    return rand_mask


def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
    """Create 4D attention mask from a 3D blocked tensor mask.

    Args:
        from_blocked_mask: 3D Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
        to_blocked_mask: 3D Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].

    Returns:
        float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                               from_block_size,  3*to_block_size].
    """
    exp_blocked_to_pad = tf.concat(
        [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2],
         to_blocked_mask[:, 3:-1]], 2)
    band_mask = tf.einsum(
        "BLQ,BLK->BLQK", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
    band_mask = tf.expand_dims(band_mask, 1)

    return band_mask
