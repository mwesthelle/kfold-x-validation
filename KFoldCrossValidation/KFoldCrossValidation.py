import random
from collections import defaultdict
from fractions import Fraction
from typing import List


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
        folds = []
        for _ in range(self.k):
            folds.append(self.generate_stratified_fold())
        remaining = []
        for klass in self.klass_idxes:
            if len(self.klass_idxes[klass]) > 0:
                remaining.extend(self.klass_idxes[klass])
        folds[-1].extend(remaining)
        test_fold_idx = random.randint(0, self.k - 1)
        test_fold = folds[test_fold_idx]


if __name__ == "__main__":
    kfold = KFoldCrossValidation("datasets/diabetes.csv", k=5)
    kfold.index_dataset()
    kfold.kfold_cross_validation()
