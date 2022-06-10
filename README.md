# Predicting gene expression from sequence using sparse-attention

This package contains the code to reproduce the work in [this report](https://currie32.github.io/s-enformer-website/).

The report is about improving upon the [Enformer model](https://www.nature.com/articles/s41592-021-01252-x) by replacing its self-attention layers with the sparse-attention layers from the [BigBird model](https://arxiv.org/abs/2007.14062). This change allows for longer DNA sequences to be used as input to the model, while using the relatively less memory. This is because memory scales quadratically with sequence length when using self-attention, compared to the linear scaling when sparse-attention is used.

The created model from this work is called sparse-enformer (s-enformer), which is also the name of this package.

## Installing s-enformer

To install the s-enformer package:
```
python3 setup.py install --user
```

## Using the package

More info about using this package is found at [s_enformer/README.md](./s_enformer).
