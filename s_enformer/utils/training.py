import functools
import os

import tensorflow as tf

from s_enformer.utils.modelling import get_shape_list


def get_dataset(organism, use_hpc, subset, num_threads=8):

    metadata = _get_metadata(organism)
    dataset = tf.data.TFRecordDataset(tfrecord_files(organism, use_hpc, subset),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_threads)

    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                          num_parallel_calls=num_threads)
    return dataset


def _get_metadata(organism):

    if organism == 'human':
        return {
            "num_targets": 5313,
            "train_seq": 34021,
            "valid_seqs": 2213,
            "test_seqs": 1937,
            "seq_length": 131072,
            "pool_width": 128,
            "crop_bp": 8192,
            "target_length": 896
        }
    elif organism == "mouse":
        return {
            "num_targets": 1643,
            "train_seq": 29295,
            "valid_seqs": 2209,
            "test_seqs": 2017,
            "seq_length": 131072,
            "pool_width": 128,
            "crop_bp": 8192,
            "target_length": 896
        }


def tfrecord_files(organism, use_hpc, subset):
    # Sort the values by int(*).
    return sorted(tf.io.gfile.glob(os.path.join(
        organism_path(organism, use_hpc), 'tfrecords', f'{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def organism_path(organism, use_hpc):

    if use_hpc:
        return os.path.join('./data', organism)

    return os.path.join('../../../../../../data/dgc21/', organism)


def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target, (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence': sequence, 'target': target}


def create_step_function(model, optimizer, head, clip_grad_norm=1.0):

    @tf.function
    def train_step(sequence_length, batch_seq, batch_target, variables_to_train):
        with tf.GradientTape() as tape:

            length = get_shape_list(batch_seq)[1]
            padding_length_left = int((sequence_length - length) // 2)
            padding_length_right = sequence_length - length - padding_length_left
            paddings = tf.constant([[0, 0, ], [padding_length_left, padding_length_right], [0, 0]])

            batch_seq = tf.pad(batch_seq, paddings, "CONSTANT")
            outputs = model.train(batch_seq)[head]

            poisson = tf.reduce_mean(tf.keras.losses.poisson(batch_target, outputs))
            loss = poisson

        gradients = tape.gradient(loss, variables_to_train, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gradients = [tf.clip_by_norm(grad, clip_grad_norm) for grad in gradients]
        ctx = tf.distribute.get_replica_context()
        gradients = ctx.all_reduce("mean", gradients)
        optimizer.apply(gradients, variables_to_train)

        return loss

    return train_step
