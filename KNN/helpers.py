import numpy as np


def normalize_data(data):
    max_element = np.max(data)
    min_element = np.min(data)
    normalized_data = np.divide(data - min_element, max_element - min_element)
    return normalized_data
