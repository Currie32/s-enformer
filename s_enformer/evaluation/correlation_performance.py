import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

from s_enformer.evaluation.assay.indices import get_assay_indices
from s_enformer.utils.modelling import get_shape_list
from s_enformer.utils.training import get_dataset


# Enformer sequence length is 393216, BigBird is half that.
# Change the sequence length for the model that you are evaluating.
SEQUENCE_LENGTH = 393216 // 2
STEPS_EVALUATION = None
USE_HPC = False
USE_PRETRAINED_ENFORMER = False


def main():

    if USE_PRETRAINED_ENFORMER:
        model = hub.load("https://tfhub.dev/deepmind/enformer/1").model

    else:
        model = tf.saved_model.load("../models/s_enformer")

    assay_indices = get_assay_indices()

    evaluate_model(model,
                   dataset=get_dataset('human', USE_HPC, 'test').batch(1).prefetch(2),
                   head='human',
                   assay_indices=assay_indices,
                   max_steps=STEPS_EVALUATION)


def evaluate_model(model, dataset, head, assay_indices, max_steps=None):

    # Store the results for each track type
    metric_DNase_ATAC = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
    metric_ChIP_histone = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
    metric_ChIP_tf = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
    metric_CAGE = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})

    @tf.function
    def predict(x):

        length = get_shape_list(x)[1]
        padding_length_left = int((SEQUENCE_LENGTH - length) // 2)
        padding_length_right = SEQUENCE_LENGTH - length - padding_length_left
        paddings = tf.constant([[0, 0, ], [padding_length_left, padding_length_right], [0, 0]])

        x = tf.pad(x, paddings, "CONSTANT")

        return model.predict_on_batch(x)[head]

    for i, batch in tqdm(enumerate(dataset)):
        if max_steps is not None and i > max_steps:
            break

        # Get the target and predicted expression levels
        target = batch['target'][0]
        sequence = predict(batch['sequence'])[0]

        # Get targets and predicted expression levels for each track type
        target_DNase_ATAC_tracks = np.array([tf.gather(i, assay_indices['DNase_ATAC']) for i in target])
        sequence_DNase_ATAC_tracks = np.array([tf.gather(i, assay_indices['DNase_ATAC']) for i in sequence])

        target_ChIP_histone_tracks = np.array([tf.gather(i, assay_indices['ChIP_histone']) for i in target])
        sequence_ChIP_histone_tracks = np.array([tf.gather(i, assay_indices['ChIP_histone']) for i in sequence])

        target_ChIP_tf_tracks = np.array([tf.gather(i, assay_indices['ChIP_tf']) for i in target])
        sequence_ChIP_tf_tracks = np.array([tf.gather(i, assay_indices['ChIP_tf']) for i in sequence])

        target_CAGE_tracks = np.array([tf.gather(i, assay_indices['CAGE']) for i in target])
        sequence_CAGE_tracks = np.array([tf.gather(i, assay_indices['CAGE']) for i in sequence])

        # Update the results for each track type
        metric_DNase_ATAC.update_state(target_DNase_ATAC_tracks, sequence_DNase_ATAC_tracks)
        metric_ChIP_histone.update_state(target_ChIP_histone_tracks, sequence_ChIP_histone_tracks)
        metric_ChIP_tf.update_state(target_ChIP_tf_tracks, sequence_ChIP_tf_tracks)
        metric_CAGE.update_state(target_CAGE_tracks, sequence_CAGE_tracks)

        if i % 10 == 0:
            print("DNase / ATAC:", {k: v.numpy().mean() for k, v in metric_DNase_ATAC.result().items()})
            print("ChIP Histone:", {k: v.numpy().mean() for k, v in metric_ChIP_histone.result().items()})
            print("ChIP TF:", {k: v.numpy().mean() for k, v in metric_ChIP_tf.result().items()})
            print("CAGE:", {k: v.numpy().mean() for k, v in metric_CAGE.result().items()})
            print()

    print("DNase / ATAC:", {k: v.numpy().mean() for k, v in metric_DNase_ATAC.result().items()})
    print("ChIP Histone:", {k: v.numpy().mean() for k, v in metric_ChIP_histone.result().items()})
    print("ChIP TF:", {k: v.numpy().mean() for k, v in metric_ChIP_tf.result().items()})
    print("CAGE:", {k: v.numpy().mean() for k, v in metric_CAGE.result().items()})


class MetricDict:
    def __init__(self, metrics):
        self._metrics = metrics

    def update_state(self, y_true, y_pred):
        for k, metric in self._metrics.items():
            metric.update_state(y_true, y_pred)

    def result(self):
        return {k: metric.result() for k, metric in self._metrics.items()}


class CorrelationStats(tf.keras.metrics.Metric):
    """Contains shared code for PearsonR and R2."""

    def __init__(self, reduce_axis=None, name='pearsonr'):
        """Pearson correlation coefficient.

        Args:
          reduce_axis: Specifies over which axis to compute the correlation (say
            (0, 1). If not specified, it will compute the correlation across the
            whole tensor.
          name: Metric name.
        """
        super(CorrelationStats, self).__init__(name=name)
        self._reduce_axis = reduce_axis
        self._shape = None  # Specified in _initialize.

    def _initialize(self, input_shape):
        # Remaining dimensions after reducing over self._reduce_axis.
        self._shape = _reduced_shape(input_shape, self._reduce_axis)

        weight_kwargs = dict(shape=self._shape, initializer='zeros')
        self._count = self.add_weight(name='count', **weight_kwargs)
        self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
        self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
        self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                                 **weight_kwargs)
        self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
        self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                                 **weight_kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state.

        Args:
          y_true: Multi-dimensional float tensor [batch, ...] containing the ground
            truth values.
          y_pred: float tensor with the same shape as y_true containing predicted
            values.
          sample_weight: 1D tensor aligned with y_true batch dimension specifying
            the weight of individual observations.
        """
        if self._shape is None:
            # Explicit initialization check.
            self._initialize(y_true.shape)

        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        self._product_sum.assign_add(
            tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

        self._true_sum.assign_add(
            tf.reduce_sum(y_true, axis=self._reduce_axis))

        self._true_squared_sum.assign_add(
            tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

        self._pred_sum.assign_add(
            tf.reduce_sum(y_pred, axis=self._reduce_axis))

        self._pred_squared_sum.assign_add(
            tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

        self._count.assign_add(
            tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

    def result(self):
        raise NotImplementedError('Must be implemented in subclasses.')

    def reset_states(self):
        if self._shape is not None:
            tf.keras.backend.batch_set_value([(v, np.zeros(self._shape))
                                              for v in self.variables])


# @title `PearsonR` and `R2` metrics
def _reduced_shape(shape, axis):
    if axis is None:
        return tf.TensorShape([])
    return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class PearsonR(CorrelationStats):
    """Pearson correlation coefficient.

    Computed as:
    ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
    """

    def __init__(self, reduce_axis=(0,), name='pearsonr'):
        """Pearson correlation coefficient.

        Args:
          reduce_axis: Specifies over which axis to compute the correlation.
          name: Metric name.
        """
        super(PearsonR, self).__init__(reduce_axis=reduce_axis,
                                       name=name)

    def result(self):
        true_mean = self._true_sum / self._count
        pred_mean = self._pred_sum / self._count

        covariance = (self._product_sum
                      - true_mean * self._pred_sum
                      - pred_mean * self._true_sum
                      + self._count * true_mean * pred_mean)

        true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
        pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
        pred_var = tf.where(
            tf.greater(pred_var, 1e-12), pred_var, np.inf * tf.ones_like(pred_var)
        )
        tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
        correlation = covariance / tp_var

        return correlation


class R2(CorrelationStats):
    """R-squared  (fraction of explained variance)."""
    def __init__(self, reduce_axis=None, name='R2'):
        """R-squared metric.

        Args:
        reduce_axis: Specifies over which axis to compute the correlation.
        name: Metric name.
        """
        super(R2, self).__init__(reduce_axis=reduce_axis, name=name)

    def result(self):
        true_mean = self._true_sum / self._count
        total = self._true_squared_sum - self._count * tf.math.square(true_mean)
        residuals = (self._pred_squared_sum - 2 * self._product_sum
                     + self._true_squared_sum)

        return tf.ones_like(residuals) - residuals / total


if __name__ == '__main__':
    main()
