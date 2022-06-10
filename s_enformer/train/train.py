import numpy as np
import sonnet as snt
import tensorflow as tf

from s_enformer.utils.extract_model_weights import create_enf_model
from s_enformer.utils import training as utils


# Training parameters
BATCH_SIZE = 5
LEARNING_RATE = 0.0001
NUM_STEPS = 100000
SAVE_MODEL = True
SEQUENCE_LENGTH = 196608
USE_HPC = False


def train_model():

    model, variables_to_train = create_enf_model('../models/enformer/')
    model.variables_to_train = variables_to_train

    learning_rate = tf.Variable(LEARNING_RATE, trainable=False, name='learning_rate')
    optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

    train_step_human = utils.create_step_function(model, optimizer, 'human')
    train_step_mouse = utils.create_step_function(model, optimizer, 'mouse')

    # Get the data
    human_dataset = utils.get_dataset('human', USE_HPC, 'train').batch(BATCH_SIZE).repeat()
    mouse_dataset = utils.get_dataset('mouse', USE_HPC, 'train').batch(BATCH_SIZE).repeat()
    human_mouse_dataset = tf.data.Dataset.zip((human_dataset, mouse_dataset)).prefetch(2)
    data_it = iter(human_mouse_dataset)

    step_i = 1
    losses = []

    while step_i <= NUM_STEPS:

        batch_human, batch_mouse = next(data_it)
        loss_human = train_step_human(SEQUENCE_LENGTH, batch_human['sequence'], batch_human['target'], variables_to_train)
        loss_mouse = train_step_mouse(SEQUENCE_LENGTH, batch_mouse['sequence'], batch_mouse['target'], variables_to_train)

        loss_human = loss_human.numpy()
        loss_mouse = loss_mouse.numpy()
        losses.append(loss_human)

        if step_i % 1 == 0:

            print({
                'step:': step_i,
                'loss_human': loss_human,
                'loss_mouse': loss_mouse,
                'loss_avg_prev_100': np.mean(losses[-100:])
            })

        if SAVE_MODEL and step_i % 100 == 0:
            tf.saved_model.save(model, "../models/s_enformer")

        step_i += 1


if __name__ == '__main__':
    train_model()
