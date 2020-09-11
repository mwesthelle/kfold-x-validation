import argparse

from KFoldCrossValidation import KFoldCrossValidation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run K-Fold Cross Validation on a KNN model for some dataset"
    )
    parser.add_argument("dataset_path")
    parser.add_argument("-k", "--k-folds", type=int, default=10)
    parser.add_argument("-r", "--repeated", type=int, default=3)
    parser.add_argument("-n", "--k-nearest-neighbors", type=int, default=3)
    parser.add_argument("-m", "--minkowski-p", type=int, default=2)
    args = parser.parse_args()
    kfold_x_validation = KFoldCrossValidation(
        args.dataset_path,
        args.k_folds,
        args.repeated,
        args.k_nearest_neighbors,
        args.minkowski_p,
    )
    kfold_x_validation.kfold_cross_validation()
