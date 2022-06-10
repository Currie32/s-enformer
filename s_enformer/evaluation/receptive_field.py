import copy
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tqdm

MODEL_ENFORMER = hub.load("https://tfhub.dev/deepmind/enformer/1").model
MODEL_BIGBIRD = tf.saved_model.load("../models/s_enformer")
SEQUENCE_LENGTH = 196608
N_EXPERIMENTS = 100
BASE_PAIRS = np.eye(4)


def measure_receptive_field():
    """
    Measure the receptive field of the Enformer and Bigbird models by
    running the mutation experiment at several locations in the sequence.
    """

    mutation_locations = np.linspace(0, SEQUENCE_LENGTH - 1, 9)
    results = {}

    for mutation_location in mutation_locations:
        mutation_location = int(mutation_location)

        results_enformer, results_bigbird = _run_mutation_experiment(mutation_location)
        results[mutation_location] = {
            'enformer': results_enformer,
            'bigbird': results_bigbird
        }

    with open('receptive_field_results.pkl', 'wb') as f:
        pickle.dump(results, f)


def _run_mutation_experiment(mutation_pos: list) -> tuple:
    """
    Measure the receptive field of both models by:
    (1) predicting the expression level on a random sequence of DNA
    (2) mutate one base pair
    (3) predict the expression level on the new sequence of DNA
    (4) measure the expression difference
    (5) repeat steps 1-4 multiple times, then take the average

    Args:
        mutation_pos (int): location in the sequence for the mutation

    Returns:
        The average change in expression level for the two models
    """

    changed_expression_enformer = []
    changed_expression_bigbird = []

    for i in tqdm.tqdm(range(N_EXPERIMENTS)):

        # Baseline results
        random_dna = BASE_PAIRS[
            np.random.choice(BASE_PAIRS.shape[0], size=196608)
        ][np.newaxis, :, :]
        # Enformer accepts a longer input sequence so the ends need to be padded
        pad_enformer = np.zeros(shape=(1, 196608 // 2, 4))
        random_dna_enformer = np.concatenate((pad_enformer, random_dna, pad_enformer), axis=1)

        baseline_enformer = MODEL_ENFORMER.predict_on_batch(random_dna_enformer)['human'][0].numpy()
        baseline_bigbird = MODEL_BIGBIRD.predict_on_batch(random_dna)['human'][0].numpy()

        # Mutation results
        mutated_dna = copy.deepcopy(random_dna)
        mutated_dna[0, mutation_pos, :] = np.roll(
            mutated_dna[0, mutation_pos, :],
            shift=np.random.randint(1, 3)
        )
        mutated_dna_enformer = np.concatenate((pad_enformer, mutated_dna, pad_enformer), axis=1)
        output_enformer = MODEL_ENFORMER.predict_on_batch(mutated_dna_enformer)['human'][0].numpy()
        output_bigbird = MODEL_BIGBIRD.predict_on_batch(mutated_dna)['human'][0].numpy()

        # Measure the difference in expression level
        difference_enformer = output_enformer - baseline_enformer
        difference_bigbird = output_bigbird - baseline_bigbird

        changed_expression_enformer.append(difference_enformer)
        changed_expression_bigbird.append(difference_bigbird)

    avg_changed_expression_enformer = list(np.mean(np.mean(np.abs(changed_expression_enformer), axis=0), axis=1))
    avg_changed_expression_bigbird = list(np.mean(np.mean(np.abs(changed_expression_bigbird), axis=0), axis=1))

    return avg_changed_expression_enformer, avg_changed_expression_bigbird


if __name__ == '__main__':
    measure_receptive_field()
