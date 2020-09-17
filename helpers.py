import numpy as np


def normalize_data(data):
    normalized_data = np.empty(data.T.shape)
    for idx, col in enumerate(data.T):
        max_element = np.max(col)
        min_element = np.min(col)
        normalized_data[idx, :] = np.divide(
            col - min_element, max_element - min_element
        )
    return normalized_data.T
