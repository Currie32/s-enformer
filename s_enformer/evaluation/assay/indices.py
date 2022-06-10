import pandas as pd


def get_assay_indices():

    assay_info = pd.read_csv('assay/assay_info.csv')

    # Create a dictionary of indices
    indices = {
        'DNase_ATAC': list(assay_info[(assay_info['assay_subtype'] == 'DNase') | (assay_info['assay_subtype'] == 'ATAC')].index),
        'ChIP_histone': list(assay_info[assay_info['assay_subtype'] == 'ChIP-Histone'].index),
        'ChIP_tf': list(assay_info[assay_info['assay_subtype'] == 'ChIP-TF'].index),
        'CAGE': list(assay_info[assay_info['assay_subtype'] == 'CAGE'].index)
    }

    return indices
