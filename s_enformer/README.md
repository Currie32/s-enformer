# Using s-enformer

You should be using a GPU to train and evaluate s-enformer, ideally with 24 GB of memory

## Download the data

Unless you are running this code from Imperial College London's HPC, and have access to the directory `neurogenomics-lab/live/Projects/enformer_bigbird/`, you'll need to download the data to train, validate, and test s-enformer. You can find this data [here](https://console.cloud.google.com/storage/browser/basenji_barnyard/data). To download the data, run:

```
gsutil cp -r gs://basenji_barnyard/data .
```

If you want to change the download location, replace `.` with your desired location.

If not already installed, you'll need to [install gsutil](https://cloud.google.com/storage/docs/gsutil_install).

You will then need to change the path to the data in [`utils/training.py#L59`](https://github.com/neurogenomics/enformer_bigbird/blob/master/s_enformer/utils/training.py#L59)

## Training the model

Inside of the [train](./train) directory, run the command below to train the model:
```
python3 train.py
```

Depending on the specifications of your machine, you might want to alter the training parameters.

The model will be saved to a `./models` directory.

## Evaluation

The [evaluation](./evaluation) directory contains all of the code to evaluate the model. All of these scripts must be run within the [evaluation](./evaluation) directory.

### [correlation_performance.py](./evaluation/correlation_performance.py)

Evaluate the model by measuring its correlation across the four genomic track types:
- DNase-Seq & ATAC-Seq
- Histone modification ChIP-Seq
- Transcription factor ChIP-Seq
- CAGE (cap analysis of gene expression)

```
python3 correlation_performance.py
```

### [correlation_between_models.py](./evaluation/correlation_between_models.py)

Measure how similar the predictions between s-enformer and enformer.

```
python3 correlation_between_models.py
```

### [memory_and_speed.py](./evaluation/memory_and_speed.py)

Measure how much memory a model uses when training and how quickly it trains.

```
python3 memory_and_speed.py
```

### [receptive_field.py](./evaluation/receptive_field.py)

Measure the size of a model's receptive field.

```
python3 receptive_field.py
```

## Figures for the report

Some of the figures for the report are created in [create_figures.ipynb](./create_figures.ipynb).
