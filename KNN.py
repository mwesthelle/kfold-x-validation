from collections import Counter
from operator import itemgetter
from typing import Iterable, List

from helpers import normalize_data

import numpy as np

from BaseModel import BaseModel


class KNNModel(BaseModel):
    """
    A K-Nearest Neighbors model

    Attributes
    ----------
    minkoswki_p : int
        Order of Minkowski distance (where 1 is Manhattan, 2 is Euclidean, and
        higher orders aren't special until Chebyshev's)

    k_neighbors : int
        Number of nearest neighbors that will vote on the new data points' classes

    Methods
    -------
    _calculate_distance(this : np.ndarray, other : np.ndarray)
        Calculate Minkowski's generalized distance between this point and the other
        point.

    load_train_data(filename: str)
        Populates a matrix of features/attributes and normalizes it according to
        min/max values

    predict(test_data: np.ndarray, k: int)
        Given a list of test data rows and k nearest neighbors, predict an outcome for
        each given data point
    """

    def __init__(self, minkowski_p: int = 2, k_neighbors: int = 3):
        self.minkoswki_p = minkowski_p
        self.features = None
        self.outcomes = []
        self.k_neighbors = k_neighbors

    def _calculate_distance(self, this, other) -> float:
        return np.sum((this - other) ** self.minkoswki_p) ** (1 / self.minkoswki_p)

    def load_train_data(self, data_iter: Iterable[List[str]]):
        n_rows = len(data_iter)
        n_columns = len(data_iter[0].split(","))
        self.features = np.empty((n_rows, n_columns - 1))
        for idx, row in enumerate(data_iter):
            values = row.split(",")
            self.features[idx] = np.array([float(val) for val in values[:-1]])
            self.outcomes.append(values[-1])
        self.features = normalize_data(self.features)

    def predict(self, test_data: Iterable[List[str]]):
        n_rows = len(test_data)
        n_columns = len(test_data[0].split(","))
        test_matrix = np.empty((n_rows, n_columns - 1))
        for idx, row in enumerate(test_data):
            values = row.split(",")
            test_matrix[idx] = np.array([float(val) for val in values[:-1]])
        test_matrix = normalize_data(test_matrix)

        predictions = np.empty(len(test_data))
        for idx, test_row in enumerate(test_matrix):
            distances = [
                (idx, self._calculate_distance(data_point, test_row))
                for idx, data_point in enumerate(self.features)
            ]
            distances.sort(key=itemgetter(1))
            k_outcomes = [
                self.outcomes[idx] for idx, _ in distances[: self.k_neighbors]
            ]
            outcome_counts = Counter(k_outcomes)
            predictions[idx] = max(outcome_counts, key=outcome_counts.get)
        return predictions
