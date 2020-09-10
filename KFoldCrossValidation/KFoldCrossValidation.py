import random
from collections import defaultdict
from fractions import Fraction
from itertools import chain
from typing import List

import numpy as np

from KNN.KNN import KNNModel


class KFoldCrossValidation:
    def __init__(self, filename: str, k: int = 10, repeated=False):
        self.filename = filename
        self.k = k
        self.repeated = repeated
        self.klass_idxes = defaultdict(
            list
        )  # Holds classes as keys and indices they occur on as values
        self._line_offsets = []

    def index_dataset(self):
        """
        Build a map from our classes to the indices they appear in, as well as a list
        of offsets for fast access.
        """
        offset: int = 0
        with open(self.filename, "rb") as dataset:
            offset += len(next(dataset))  # skip header and set offset
            for idx, row in enumerate(dataset):
                self._line_offsets.append(offset)
                offset += len(row)
                values = row.decode("utf-8").strip().split(",")
                self.klass_idxes[values[-1]].append(idx)

    def generate_stratified_fold(self) -> List[int]:
        """
        Generate a stratified fold by sampling our index map without repetition. The
        fold is represented by a list of indices.
        """
        klass_proportions = {}
        fold_size = len(self._line_offsets) // self.k
        fold = []
        for klass in self.klass_idxes:
            proportion = Fraction(
                numerator=len(self.klass_idxes[klass]),
                denominator=len(self._line_offsets),
            )
            klass_proportions[klass] = proportion
            random.shuffle(self.klass_idxes[klass])
        for _ in range(fold_size):
            # Choose a random class using the class proportions as weights for the
            # random draw
            chosen_klass = random.choices(
                list(klass_proportions.keys()),
                weights=list(klass_proportions.values()),
                k=1,
            )[0]
            chosen_idx = 0
            try:
                chosen_idx = self.klass_idxes[chosen_klass].pop()
            except IndexError:
                del self.klass_idxes[chosen_klass]
                del klass_proportions[chosen_klass]
                chosen_klass = random.choices(
                    list(klass_proportions.keys()),
                    weights=list(klass_proportions.values()),
                )[0]
                chosen_idx = self.klass_idxes[chosen_klass].pop()
            finally:
                fold.append(chosen_idx)
        return fold

    def kfold_cross_validation(self):
        with open(self.filename, "rb") as dataset:
            folds = []
            for _ in range(self.k):
                fold_rows = []
                for idx in self.generate_stratified_fold():
                    dataset.seek(self._line_offsets[idx])
                    fold_rows.append(dataset.readline().decode("utf-8").strip())
                folds.append(fold_rows)

            remaining_idxs = []
            for klass in self.klass_idxes:
                if len(idxes := self.klass_idxes[klass]) > 0:
                    remaining_idxs.extend(idxes)
            remaining_data = []
            for idx in remaining_idxs:
                dataset.seek(self._line_offsets[idx])
                remaining_data.append(dataset.readline().decode("utf-8").strip())
            folds[-1].extend(remaining_data)

            accuracies = []
            fold_idxes = list(range(len(folds)))
            random.shuffle(fold_idxes)
            for _ in range(self.k):
                test_fold_idx = fold_idxes.pop()

                test_outcomes = np.array([int(t[-1]) for t in folds[test_fold_idx]])
                knn_model = KNNModel(minkowski_p=2, k_neighbors=6)
                train_folds = list(
                    chain(*(folds[:test_fold_idx] + folds[test_fold_idx + 1 :]))
                )
                knn_model.load_train_data(train_folds)
                predictions = knn_model.predict(folds[test_fold_idx])

                accuracies.append(np.mean(test_outcomes == predictions))

            print(np.mean(accuracies))
            print(np.std(accuracies))


if __name__ == "__main__":
    kfold = KFoldCrossValidation("datasets/diabetes.csv", k=10)
    kfold.index_dataset()
    kfold.kfold_cross_validation()
