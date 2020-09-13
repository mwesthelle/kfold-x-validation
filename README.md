[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Overview
========
This project implements a KNN model and k-fold cross-validation to measure its
performance.

## How to run
This project uses [Docker](https://www.docker.com/get-started).

Simply run `docker build -t kfold . && docker run -p 8080:8080 kfold`, and use the URL in the output to access
the `K-Fold Cross Validation Experiments.ipynb` Jupyter notebook.
