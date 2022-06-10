import sys

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.load import Loader
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.util import nest
import tensorflow as tf

from s_enformer import enformer


CHANNELS = 1536
TENSOR_LENGTH = CHANNELS * 128


def _get_loader(export_dir, tags=None, options=None):
    """
    Loader implementation.

    Custom to get weights from tf.hub model.

    Created by William Beardall <william.beardall15@imperial.ac.uk>
    """
    options = options or load_options.LoadOptions()
    if tags is not None and not isinstance(tags, set):
        # Supports e.g. tags=SERVING and tags=[SERVING]. Sets aren't considered
        # sequences for nest.flatten, so we put those through as-is.
        tags = nest.flatten(tags)
    saved_model_proto, debug_info = (
        loader_impl.parse_saved_model_with_debug_info(export_dir))

    if (len(saved_model_proto.meta_graphs) == 1
            and saved_model_proto.meta_graphs[0].HasField("object_graph_def")):
        meta_graph_def = saved_model_proto.meta_graphs[0]
        # tensor_content field contains raw bytes in litle endian format
        # which causes problems when loaded on big-endian systems
        # requiring byteswap
        if sys.byteorder == "big":
            saved_model_utils.swap_function_tensor_content(meta_graph_def, "little", "big")

        if (tags is not None and set(tags) != set(meta_graph_def.meta_info_def.tags)):
            raise ValueError(
                ("The SavedModel at {} has one MetaGraph with tags {}, but got an "
                 "incompatible argument tags={} to tf.saved_model.load. You may omit "
                 "it, pass 'None', or pass matching tags.")
                .format(export_dir, meta_graph_def.meta_info_def.tags, tags))
        object_graph_proto = meta_graph_def.object_graph_def

        ckpt_options = checkpoint_options.CheckpointOptions(
            experimental_io_device=options.experimental_io_device)
        with ops.init_scope():
            try:
                save_options = lambda: None
                save_options.experimental_skip_checkpoint = None
                save_options.allow_partial_checkpoint = None
                loader = Loader(object_graph_proto, saved_model_proto, export_dir,
                                ckpt_options, save_options, filters=None)
            except errors.NotFoundError as err:
                raise FileNotFoundError(
                    str(err) + "\n If trying to load on a different device from the "
                    "computational device, consider using setting the "
                    "`experimental_io_device` option on tf.saved_model.LoadOptions "
                    "to the io_device such as '/job:localhost'."
                )
            return loader


def create_enf_model(path: str):
    """
    Lift weights from pre-trained Enformer model available in tf.hub
    and move to tf.keras model so that it can be used for fine-tuning.
    Includes tests to ensure model performs _close to_ pre-trained
    model's predictions.

    Requires 16GB+ of RAM to run

    Arguements:
    path: Path to enformer model scripts.
    """

    loader = _get_loader(path)
    variable_nodes = [n for n in loader._nodes if isinstance(n, tf.Variable)]
    # first one isn't a tf.Variable from the model, remove it
    variable_nodes = variable_nodes[1:]

    # create model with initialised weights
    model = enformer.Enformer(channels=CHANNELS,
                              num_heads=8,
                              num_transformer_layers=11,
                              pooling_type='attention')

    # have to pass data through model to initialise all the weights
    # note this will require quite a bit of RAM
    # outputs = model(tf.zeros([1, 196_608, 4], tf.float32), is_training=True)
    # {'sequence': (TensorShape([1, 131072, 4]), tf.float32), 'target': (TensorShape([1, 896, 5313]), tf.float32)}
    g1 = tf.random.Generator.from_seed(1)
    rand_batch = {'sequence': tf.zeros([1, TENSOR_LENGTH, 4], tf.float32), 'target': g1.normal(shape=[1, 896, 5313])}
    _ = model(rand_batch['sequence'], is_training=True)

    # now let's make sure the number of weights is the same
    init_enf_names = [n.name for n in model.variables]
    extrac_weight_names = [n.name for n in variable_nodes]

    assert len(extrac_weight_names) == len(init_enf_names), "Number of weights don't match"

    # order and naming needs to be updated, once they match we can update by name
    for i in range(len(variable_nodes)):
        variable_nodes[i] = tf.Variable(variable_nodes[i].value(), name='enformer' + extrac_weight_names[i][5:])

    # update names
    extrac_weight_names = [n.name for n in variable_nodes]

    # model duplicates the naming convention after second '/', do the same
    to_dup = ['/final_pointwise/', '/stem/', '/conv_tower/', '/transformer/', '/head_human/',
              '/head_mouse/', '/mha/', '_layer/linear/', '/normalization/', '/batch_norm/layer_norm/',
              '/conv_block/', '/pointwise_conv_block/',
              '/downres/', '/cross_replica_batch_norm/',
              '/exponential_moving_average/',
              '/normalization/layer_norm/',
              '/pooling/softmax_pooling/',
              '/conv_block/conv_block/batch_norm/batch_norm/',
              '/pointwise_conv_block/pointwise_conv_block/batch_norm/batch_norm/',
              'enformer/',
              '/mlp/'] + ['/transformer_block_' + str(i) + '/'
                          for i in range(11)] + ['/downres_block_' + str(i) + '/' for i in range(11)]
    dup_with = ['/final_pointwise/final_pointwise/', '/stem/stem/', '/conv_tower/conv_tower/',
                '/transformer/transformer/', '/head_human/head_human/',
                '/head_mouse/head_mouse/', '/mha/mha/', '_layer/', '/batch_norm/', '/layer_norm/',
                '/conv_block/conv_block/', '/pointwise_conv_block/pointwise_conv_block/',
                '/conv_tower/conv_tower/', '/batch_norm/',
                '/moving_mean/',
                '/layer_norm/',
                '/softmax_pooling/',
                '/conv_block/conv_block/batch_norm/',
                '/pointwise_conv_block/pointwise_conv_block/batch_norm/',
                '',
                '/mlp/mlp/'] + ['/transformer_block_' + str(i) + '/transformer_block_' + str(i) + '/'
                                for i in range(11)] + ['/conv_tower_block_' + str(i) + '/conv_tower_block_' + str(i) + '/'
                                                       for i in range(11)]
    # also remove ending of :0:0 with :0 - happens when you update the name
    end_long = ':0:0'
    end_short = ''

    for i in range(len(variable_nodes)):
        new_name = variable_nodes[i].name
        for j, dup_name in enumerate(to_dup):
            if (dup_name in new_name):
                new_name = new_name.replace(dup_name, dup_with[j])
                # attention_k needs to match transformer_block_k
                if('/transformer_block_' in dup_name):
                    # also replace...
                    if('/multihead_attention/' in new_name):
                        new_name = new_name.replace('/multihead_attention/',
                                                    '/attention_' + dup_name.rsplit('_', 1)[1])

        new_name = new_name.replace(end_long, end_short)
        variable_nodes[i] = tf.Variable(variable_nodes[i].value(), name=new_name)

    # update names
    extrac_weight_names = [n.name for n in variable_nodes]

    # add in shape too since names aren't unique in either
    init_enf_names = [n.name + ' : ' + str(n.shape) for n in model.variables]
    init_enf_names = [i.lstrip('enformer/') for i in init_enf_names]
    extrac_weight_names = [n.name + ' : ' + str(n.shape) for n in variable_nodes]
    extrac_weight_names = [e for e in extrac_weight_names if 'attention' not in e]

    assert len(list(set(extrac_weight_names) - set(init_enf_names))) == 0, "Not all extracted weights are in initialised"

    # issue is moving_variance is missing from the extracted weights and is named moving_mean instead
    # can't tell which ones should be variance and which mean as shape is the same so let's just guess and test
    # Approach: rename the second set of moving_mean to moving_variance (in chronological order)
    init_enf_names = [n.name for n in model.variables]
    extrac_weight_names = [n.name for n in variable_nodes]

    # find those to be updated
    # [n.name+' : '+str(i) for i,n in enumerate(variable_nodes)]

    # Those to change
    # 'enformer/trunk/final_pointwise/final_pointwise/conv_block/conv_block/batch_norm/moving_mean/counter:0 : 36',
    # 'enformer/trunk/final_pointwise/final_pointwise/conv_block/conv_block/batch_norm/moving_mean/hidden:0 : 37',
    # 'enformer/trunk/final_pointwise/final_pointwise/conv_block/conv_block/batch_norm/moving_mean/average:0 : 38',
    # 'enformer/trunk/stem/stem/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/counter:0 : 42',
    # 'enformer/trunk/stem/stem/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/hidden:0 : 43',
    # 'enformer/trunk/stem/stem/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/average:0 : 44',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_0/conv_tower_block_0/conv_block/conv_block/batch_norm/moving_mean/counter:0 : 138',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_0/conv_tower_block_0/conv_block/conv_block/batch_norm/moving_mean/hidden:0 : 139',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_0/conv_tower_block_0/conv_block/conv_block/batch_norm/moving_mean/average:0 : 140',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_1/conv_tower_block_1/conv_block/conv_block/batch_norm/moving_mean/counter:0 : 146',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_1/conv_tower_block_1/conv_block/conv_block/batch_norm/moving_mean/hidden:0 : 147',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_1/conv_tower_block_1/conv_block/conv_block/batch_norm/moving_mean/average:0 : 148',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_2/conv_tower_block_2/conv_block/conv_block/batch_norm/moving_mean/counter:0 : 154',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_2/conv_tower_block_2/conv_block/conv_block/batch_norm/moving_mean/hidden:0 : 155',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_2/conv_tower_block_2/conv_block/conv_block/batch_norm/moving_mean/average:0 : 156',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_3/conv_tower_block_3/conv_block/conv_block/batch_norm/moving_mean/counter:0 : 162',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_3/conv_tower_block_3/conv_block/conv_block/batch_norm/moving_mean/hidden:0 : 163',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_3/conv_tower_block_3/conv_block/conv_block/batch_norm/moving_mean/average:0 : 164',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_4/conv_tower_block_4/conv_block/conv_block/batch_norm/moving_mean/counter:0 : 170',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_4/conv_tower_block_4/conv_block/conv_block/batch_norm/moving_mean/hidden:0 : 171',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_4/conv_tower_block_4/conv_block/conv_block/batch_norm/moving_mean/average:0 : 172',
    #  'enformer/trunk/conv_tower/conv_tower/conv_tower_block_5/conv_tower_block_5/conv_block/conv_block/batch_norm/moving_mean/counter:0 : 178',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_5/conv_tower_block_5/conv_block/conv_block/batch_norm/moving_mean/hidden:0 : 179',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_5/conv_tower_block_5/conv_block/conv_block/batch_norm/moving_mean/average:0 : 180',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_0/conv_tower_block_0/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/counter:0 : 230',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_0/conv_tower_block_0/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/hidden:0 : 231',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_0/conv_tower_block_0/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/average:0 : 232',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_1/conv_tower_block_1/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/counter:0 : 236',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_1/conv_tower_block_1/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/hidden:0 : 237',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_1/conv_tower_block_1/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/average:0 : 238',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_2/conv_tower_block_2/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/counter:0 : 242',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_2/conv_tower_block_2/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/hidden:0 : 243',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_2/conv_tower_block_2/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/average:0 : 244',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_3/conv_tower_block_3/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/counter:0 : 248',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_3/conv_tower_block_3/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/hidden:0 : 249',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_3/conv_tower_block_3/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/average:0 : 250',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_4/conv_tower_block_4/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/counter:0 : 254',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_4/conv_tower_block_4/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/hidden:0 : 255',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_4/conv_tower_block_4/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/average:0 : 256',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_5/conv_tower_block_5/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/counter:0 : 260',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_5/conv_tower_block_5/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/hidden:0 : 261',
    # 'enformer/trunk/conv_tower/conv_tower/conv_tower_block_5/conv_tower_block_5/pointwise_conv_block/pointwise_conv_block/batch_norm/moving_mean/average:0 : 262',
    indices_change_var = [36, 37, 38, 42, 43, 44, 138, 139, 140, 146, 147, 148, 154, 155, 156, 162, 163, 164, 170, 171,
                          172, 178, 179, 180, 230, 231, 232, 236, 237, 238, 242, 243, 244, 248, 249, 250, 254, 255, 256,
                          260, 261, 262]
    # also remove ending of :0:0 with :0 - happens when you update the name
    end_long = ':0'
    end_short = ''
    for i in indices_change_var:
        new_name = variable_nodes[i].name
        new_name = new_name.replace('moving_mean', 'moving_variance')
        new_name = new_name.replace(end_long, end_short)
        variable_nodes[i] = tf.Variable(variable_nodes[i].value(), name=new_name)

    # update names
    extrac_weight_names = [n.name for n in variable_nodes]

    # check they are matching
    extrac_weight_names = [n.name + ' : ' + str(n.shape) for n in variable_nodes]
    extrac_weight_names = [e for e in extrac_weight_names if 'attention' not in e]
    init_enf_names = [n.name + ' : ' + str(n.shape) for n in model.variables]
    init_enf_names = [i.lstrip('enformer/') for i in init_enf_names]
    init_enf_names = [i for i in init_enf_names if not any([j in i for j in ['dense', 'k_layer', 'q_layer', 'v_layer']])]

    assert len(list(set(extrac_weight_names) - set(init_enf_names))) == 0, "Not all extracted weights are in initialised"
    assert len(list(set(init_enf_names) - set(extrac_weight_names))) == 0, "Not all initialised weights are in extracted"

    extrac_weight_names = [n.name + ' : ' + str(n.shape) for n in variable_nodes]
    attention_variables = []

    # great so now we can update the initiated model with the weights from the pre-trained
    # get index where there is a match
    for index, variable in enumerate(model.variables):
        weight_name = variable.name.lstrip('enformer/') + ' : ' + str(variable.shape)

        if (weight_name in init_enf_names and 'mha' not in weight_name):
            extrac_i = extrac_weight_names.index(weight_name)

            model.variables[index].assign(variable_nodes[extrac_i])
        elif any([i in weight_name for i in ['dense', 'k_layer', 'q_layer', 'v_layer', 'mha']]):
            attention_variables.append(variable)

    return model, attention_variables
