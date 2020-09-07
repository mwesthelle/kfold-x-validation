import csv
from collections import Counter
from operator import itemgetter
from typing import List

import numpy as np


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
        Given a test data point and k neighbors, predict its outcome
    """

    def __init__(self, minkowski_p: int = 2):
        self.minkoswki_p = minkowski_p
        self.features: List[List] = []
        self.outcomes: List[int] = []

    def load_data(self, filename: str):
        with open(filename, "r") as data_file:
            reader = csv.DictReader(data_file)
            for row in reader:
                values = list(row.values())
                self.features.append(
                    np.array([float(i) for i in values[:-1]], dtype=np.float)
                )
                self.outcomes.append(int(values[-1]))
            self._normalize_data()

    def _normalize_data(self):
        max_element = 0
        min_element = np.inf
        for feat_row in self.features:
            if (max_row := np.max(feat_row)) > max_element:
                max_element = max_row
            if (min_row := np.min(feat_row)) < min_element:
                min_element = min_row
        for idx, feat_row in enumerate(self.features):
            self.features[idx] = np.divide(
                feat_row - min_element, max_element - min_element
            )

    @staticmethod
    def _normalize_test_data_point(test_data_point):
        max_element = np.max(test_data_point)
        min_element = np.min(test_data_point)
        val = np.divide(test_data_point - min_element, max_element - min_element)
        return val

    def _calculate_distance(self, this, other) -> float:
        return np.sum((this - other) ** self.minkoswki_p) ** (1 / self.minkoswki_p)

    def predict(self, test_data: List[float], k: int = 1):
        test_data = KNNModel()._normalize_test_data_point(test_data)
        distances = [
            (idx, self._calculate_distance(data_point, test_data))
            for idx, data_point in enumerate(self.features)
        ]
        distances.sort(key=itemgetter(1))
        k_outcomes = [self.outcomes[idx] for idx, _ in distances[:k]]
        outcome_counts = Counter(k_outcomes)
        return max(outcome_counts, key=outcome_counts.get)
