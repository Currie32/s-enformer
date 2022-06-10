import pickle

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

from s_enformer.evaluation.assay.indices import get_assay_indices
from s_enformer.utils.evaluation import MetricDict, PearsonR
from s_enformer.utils.modelling import get_shape_list
from s_enformer.utils.training import get_dataset


def measure_correlations():

    enformer = hub.load("https://tfhub.dev/deepmind/enformer/1").model
    bigbird = tf.saved_model.load("../models/s_enformer")

    assay_indices = get_assay_indices()

    correlate_models(enformer,
                     bigbird,
                     dataset=get_dataset('human', False, 'test').batch(1).prefetch(2),
                     assay_indices=assay_indices)


def correlate_models(enformer, bigbird, dataset, assay_indices):

    correlations = {
        'DNase_ATAC': {'enformer': [], 'bigbird': []},
        'ChIP_histone': {'enformer': [], 'bigbird': []},
        'ChIP_tf': {'enformer': [], 'bigbird': []},
        'CAGE': {'enformer': [], 'bigbird': []}
    }
    genomic_track_types = ['DNase_ATAC', 'ChIP_histone', 'ChIP_tf', 'CAGE']

    for i, batch in tqdm(enumerate(dataset)):

        # Get the target and predicted expression levels
        target = batch['target'][0]
        pred_enformer = predict(batch['sequence'], enformer, 393216)[0]
        pred_bigbird = predict(batch['sequence'], bigbird, 393216 // 2)[0]

        for track in genomic_track_types:

            # Store correlation results for each batch
            metric_enformer = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
            metric_bigbird = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})

            # Get targets and predicted expression levels for each track type
            target_tracks = np.array([tf.gather(i, assay_indices[track]) for i in target])
            pred_tracks_enformer = np.array([tf.gather(i, assay_indices[track]) for i in pred_enformer])
            pred_tracks_bigbird = np.array([tf.gather(i, assay_indices[track]) for i in pred_bigbird])
            metric_enformer.update_state(target_tracks, pred_tracks_enformer)
            metric_bigbird.update_state(target_tracks, pred_tracks_bigbird)
            correlations[track]['enformer'].append(list(v.numpy() for k, v in metric_enformer.result().items())[0])
            correlations[track]['bigbird'].append(list(v.numpy() for k, v in metric_bigbird.result().items())[0])

        # Save the correlation results after each testing sequence
        with open('correlation_results.pkl', 'wb') as f:
            pickle.dump(correlations, f)


@tf.function
def predict(x, model, sequence_length):

    length = get_shape_list(x)[1]
    padding_length_left = int((sequence_length - length) // 2)
    padding_length_right = sequence_length - length - padding_length_left
    paddings = tf.constant([[0, 0, ], [padding_length_left, padding_length_right], [0, 0]])

    x = tf.pad(x, paddings, "CONSTANT")

    return model.predict_on_batch(x)['human']


if __name__ == '__main__':
    measure_correlations()
