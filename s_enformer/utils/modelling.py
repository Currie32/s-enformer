"""Helper and utility functions."""

import tensorflow.compat.v2 as tf


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if not tf.executing_eagerly() and name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]

    return shape


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if not tf.executing_eagerly() and name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, int):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.compat.v1.get_variable_scope().name
        raise ValueError(
            "For the tensor `{}` in scope `{}`, the actual rank "
            "`{}` (shape = {}) is not equal to the expected rank `{}`".format(
                name, scope_name, actual_rank, str(tensor.shape),
                str(expected_rank)
            )
        )


############################### DENSE LAYERS ###################################


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""

    return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


class Dense3dLayer(tf.keras.layers.Layer):
    """A dense layer with 3D kernel."""

    def __init__(self,
                 num_attention_heads,
                 size_per_head,
                 initializer,
                 activation,
                 name=None,
                 head_first=False,
                 use_bias=True):
        """Constructor for dense layer with 3D kernel.

        Args:
            num_attention_heads: The size of output dimension.
            size_per_head: The size per attention head.
            initializer: Kernel initializer.
            activation: Actication function.
            name: The name scope of this layer.
            head_first: Whether to output head dimension before or after sequence dim.
            use_bias: Whether the layer uses a bias vector.
        """
        super(Dense3dLayer, self).__init__(name=name)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.initializer = initializer
        self.activation = activation
        self.head_first = head_first
        self.use_bias = use_bias

        with tf.compat.v1.variable_scope(name):
            hidden_size = self.num_attention_heads * self.size_per_head

            self.w = tf.compat.v1.get_variable(
                name="kernel",
                shape=[hidden_size, hidden_size],
                initializer=self.initializer)

            if self.use_bias:
                self.b = tf.compat.v1.get_variable(
                    name="bias",
                    shape=[hidden_size],
                    initializer=tf.zeros_initializer())
            else:
                self.b = None

    def call(self, input_tensor):
        """Constructor for dense layer with 3D kernel.

        Args:
            input_tensor: float Tensor of shape [batch, seq_length, hidden_size].

        Returns:
            float logits Tensor.
        """
        hidden_size = self.num_attention_heads * self.size_per_head

        reshape_w = tf.reshape(
            self.w, [hidden_size, self.num_attention_heads, self.size_per_head])

        if self.head_first:
            ret = tf.einsum("abc,cde->adbe", input_tensor, reshape_w)
        else:
            ret = tf.einsum("abc,cde->abde", input_tensor, reshape_w)

        if self.use_bias:
            if self.head_first:
                reshape_b = tf.reshape(
                    self.b, [1, self.num_attention_heads, 1, self.size_per_head])
            else:
                reshape_b = tf.reshape(
                    self.b, [self.num_attention_heads, self.size_per_head])
            ret += reshape_b

        if self.activation is not None:
            return self.activation(ret)
        else:
            return ret


class Dense3dProjLayer(tf.keras.layers.Layer):
    """A dense layer with 3D kernel for projection."""

    def __init__(self,
                 num_attention_heads,
                 size_per_head,
                 initializer,
                 activation,
                 name=None,
                 use_bias=True):
        """Constructor for dense layer with 3D kernel for projection.

        Args:
            num_attention_heads: The size of output dimension.
            size_per_head: The size per attention head.
            initializer: Kernel initializer.
            activation: Actication function.
            name: The name scope of this layer.
            use_bias: Whether the layer uses a bias vector.
        """
        super(Dense3dProjLayer, self).__init__(name=name)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.initializer = initializer
        self.activation = activation
        self.use_bias = use_bias

        with tf.compat.v1.variable_scope(name):
            hidden_size = self.num_attention_heads * self.size_per_head
            self.w = tf.compat.v1.get_variable(
                name="kernel",
                shape=[hidden_size, hidden_size],
                initializer=self.initializer)

            if self.use_bias:
                self.b = tf.compat.v1.get_variable(
                    name="bias",
                    shape=[hidden_size],
                    initializer=tf.zeros_initializer())
            else:
                self.b = None

    def call(self, input_tensor):
        """Constructor for dense layer with 3D kernel for projection.

        Args:
            input_tensor: float Tensor of shape [batch,from_seq_length,
                num_attention_heads, size_per_head].

        Returns:
            float logits Tensor.
        """
        hidden_size = self.num_attention_heads * self.size_per_head
        reshape_w = tf.reshape(
            self.w, [self.num_attention_heads, self.size_per_head, hidden_size])
        ret = tf.einsum("BFNH,NHD->BFD", input_tensor, reshape_w)

        if self.use_bias:
            ret += self.b

        if self.activation is not None:
            return self.activation(ret)
        else:
            return ret
