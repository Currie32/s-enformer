import time

import numpy as np
import sonnet as snt
import tensorflow as tf

from s_enformer import enformer
from s_enformer.utils import training as utils


# Model parameters
CHANNELS = 1536
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 11
POOLING_TYPE = 'attention'

# Training parameters
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_STEPS = 10
SEQUENCE_LENGTH = 196608


def measure_memory_and_speed():

    model = enformer.Enformer(
        channels=CHANNELS,
        num_heads=NUM_HEADS,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        pooling_type='attention'
    )
    variables_to_train = model.trainable_variables

    learning_rate = tf.Variable(LEARNING_RATE, trainable=False, name='learning_rate')
    optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

    train_step_human = utils.create_step_function(model, optimizer, 'human')
    train_step_mouse = utils.create_step_function(model, optimizer, 'mouse')

    # Get the data
    human_dataset = utils.get_dataset('human', False, 'train').batch(BATCH_SIZE).repeat()
    mouse_dataset = utils.get_dataset('mouse', False, 'train').batch(BATCH_SIZE).repeat()
    human_mouse_dataset = tf.data.Dataset.zip((human_dataset, mouse_dataset)).prefetch(2)
    data_it = iter(human_mouse_dataset)

    step_i = 1
    memory_usage_all = []
    start = time.time()

    while step_i <= NUM_STEPS:

        batch_human, batch_mouse = next(data_it)

        tf.config.experimental.reset_memory_stats('GPU:0')

        loss_human = train_step_human(SEQUENCE_LENGTH, batch_human['sequence'], batch_human['target'], variables_to_train)
        loss_mouse = train_step_mouse(SEQUENCE_LENGTH, batch_mouse['sequence'], batch_mouse['target'], variables_to_train)

        memory_usage = tf.config.experimental.get_memory_info('GPU:0')['peak']
        memory_usage_all.append(memory_usage)

        print(f"GPU usage, step {step_i}:", memory_usage)

        loss_human = loss_human.numpy()
        loss_mouse = loss_mouse.numpy()

        step_i += 1

    end = time.time()
    total_time = end - start
    avg_step_time = round(total_time / NUM_STEPS, 2)
    avg_memory_usage = round(np.mean(memory_usage_all) / 10**9, 2)
    print(f"Avg step time: {avg_step_time}s")
    print(f"Avg memory usage: {avg_memory_usage}GB")


if __name__ == '__main__':
    measure_memory_and_speed()
