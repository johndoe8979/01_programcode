import numpy as np

#############################################################################
# Function List
#############################################################################


def sigmoid_func(x):
    return 1 / (1 + np.exp(x))


def softmax_func(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    # オーバーフロー対策
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
