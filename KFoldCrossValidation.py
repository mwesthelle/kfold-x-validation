import random
from collections import defaultdict
from fractions import Fraction
from itertools import chain
from typing import List

from KNN import KNNModel

from Metrics import f_measure, precision

import numpy as np


class KFoldCrossValidation:
    def __init__(
        self,
        filename: str,
        k_folds: int = 10,
        r: int = 1,
        k_nearest_neighbors: int = 3,
        minkowski_p: int = 2,
    ):
        self.filename = filename
        self.k_folds = k_folds
        self.r = r
        self.klass_idxes = defaultdict(
            list
        )  # Holds classes as keys and indices they occur on as values
        self._line_offsets = []
        self.k_nearest_neighbors = k_nearest_neighbors
        self.minkowski_p = minkowski_p

    def index_dataset(self):
        """
        Build a map from our classes to the indices they appear in, as well as a list
        of offsets for fast access.
        """
        offset: int = 0
        self._line_offsets.clear()
        with open(self.filename, "rb") as dataset:
            dataset.seek(0)
            offset += len(next(dataset))  # skip header and set offset
            for idx, row in enumerate(dataset):
                self._line_offsets.append(offset)
                offset += len(row)
                values = row.decode("utf-8").strip().split(",")
                self.klass_idxes[values[-1]].append(idx)

    def generate_stratified_fold(self, seed: int = 1) -> List[int]:
        """
        Generate a stratified fold by sampling our index map without repetition. The
        fold is represented by a list of indices.
        """
        klass_proportions = {}
        fold_size = len(self._line_offsets) // self.k_folds
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
            repeated_precisions_mean = []
            repeated_precisions_std = []
            repeated_f1_mean = []
            repeated_f1_std = []
            for i_repetition in range(self.r):
                print(f"Repetition {i_repetition + 1}")
                print("============")
                random.seed(i_repetition * 3)
                self.index_dataset()
                folds = []
                for _ in range(self.k_folds):
                    fold_rows = []
                    for idx in self.generate_stratified_fold():
                        dataset.seek(self._line_offsets[idx])
                        fold_rows.append(dataset.readline().decode("utf-8").strip())
                    folds.append(fold_rows)

                remaining_idxs = []
                for klass in self.klass_idxes:
                    if len(idxes := self.klass_idxes[klass]) > 0:
                        remaining_idxs.extend(idxes)
                self.klass_idxes.clear()
                remaining_data = []
                for idx in remaining_idxs:
                    dataset.seek(self._line_offsets[idx])
                    remaining_data.append(dataset.readline().decode("utf-8").strip())
                folds[-1].extend(remaining_data)

                precisions = []
                f1_scores = []
                fold_idxes = list(range(len(folds)))
                random.shuffle(fold_idxes)
                knn_model = KNNModel(
                    minkowski_p=self.minkowski_p, k_neighbors=self.k_nearest_neighbors
                )
                for i_fold in range(self.k_folds):
                    test_fold_idx = fold_idxes.pop()

                    test_outcomes = np.array([int(t[-1]) for t in folds[test_fold_idx]])
                    train_folds = list(
                        chain(*(folds[:test_fold_idx] + folds[test_fold_idx + 1 :]))
                    )
                    knn_model.load_train_data(train_folds)
                    predictions = knn_model.predict(folds[test_fold_idx])

                    precisions.append(prec := precision(predictions, test_outcomes))
                    f1_scores.append(f1 := f_measure(predictions, test_outcomes))
                    print(f"Fold number {i_fold + 1}:")
                    print(f"Precision: {prec:.2f}")
                    print(f"F1-score: {f1:.2f}")
                    print("----------")

                print()
                print("============")
                repeated_precisions_mean.append(prec_mean := np.mean(precisions))
                repeated_precisions_std.append(prec_std := np.std(precisions))
                repeated_f1_mean.append(f1_mean := np.mean(f1_scores))
                repeated_f1_std.append(f1_std := np.std(f1_scores))
                print(
                    f"Mean precision of repetition {i_repetition + 1}: {prec_mean:.2f}"
                )
                print(
                    "Precision standard deviation of repetition "
                    f"{i_repetition + 1}: {prec_std:.2f}"
                )
                print(f"Mean F1 score of repetition {i_repetition + 1}: {f1_mean:.2f}")
                print(
                    "F1 score standard deviation of repetition "
                    f"{i_repetition + 1}: {f1_std:.2f}"
                )

            print("~~~~~~~~~~")
            print(f"Precision mean: {np.mean(repeated_precisions_mean):.2f}")
            print(
                f"Precision standard deviation: {np.mean(repeated_precisions_std):.2f}"
            )
            print(f"f1 mean: {np.mean(repeated_f1_mean):.2f}")
            print(f"f1 standard deviation: {np.mean(repeated_f1_std):.2f}")
