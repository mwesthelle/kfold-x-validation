from collections import Counter
from operator import itemgetter
from typing import Iterable, List

import numpy as np

from .helpers import normalize_data


class KNNModel:
    """
    A K-Nearest Neighbors model

    Attributes
    ----------
    minkoswki_p : int
        Order of Minkowski distance (where 1 is Manhattan, 2 is Euclidean, and
        higher orders aren't special until Chebyshev's)

    Methods
    -------
    load_data(filename: str)
        Loads a .csv file into memory to train a KNN model on it

    predict(test_data: np.ndarray, k: int)
        Given a list of test data and k neighbors, predict an outcome for each given
        data point
    """

    def __init__(self, minkowski_p: int = 2, k_neighbors: int = 3):
        self.minkoswki_p = minkowski_p
        self.features = None
        self.outcomes = None
        self.k_neighbors = k_neighbors

    def load_train_data(self, data_iter: Iterable[List[str]]):
        n_rows = len(data_iter)
        n_columns = len(data_iter[0].split(","))
        self.features = np.empty((n_rows, n_columns - 1))
        self.outcomes = np.empty((n_rows), dtype=np.dtype("u1"))
        for idx, row in enumerate(data_iter):
            values = row.split(",")
            self.features[idx] = np.array([float(val) for val in values[:-1]])
            self.outcomes[idx] = int(values[-1])
        self.features = normalize_data(self.features)

    def _calculate_distance(self, this, other) -> float:
        return np.sum((this - other) ** self.minkoswki_p) ** (1 / self.minkoswki_p)

    def predict(self, test_data: Iterable[List[str]]):
        n_rows = len(test_data)
        n_columns = len(test_data[0].split(","))
        test_matrix = np.empty((n_rows, n_columns - 1))
        for idx, row in enumerate(test_data):
            values = row.split(",")
            test_matrix[idx] = np.array([float(val) for val in values[:-1]])
        test_matrix = normalize_data(test_matrix)
        # Pre-allocate memory for outcomes list
        predictions = np.empty(len(test_data))
        for idx, test_row in enumerate(test_matrix):
            distances = [
                (idx, self._calculate_distance(data_point, test_row))
                for idx, data_point in enumerate(self.features)
            ]
            distances.sort(key=itemgetter(1))
            assert all([type(v) == tuple for v in distances])
            k_outcomes = [
                self.outcomes[idx] for idx, _ in distances[: self.k_neighbors]
            ]
            outcome_counts = Counter(k_outcomes)
            predictions[idx] = max(outcome_counts, key=outcome_counts.get)
        return predictions
