import numpy as np


def distances_parallel(all_samples: np.ndarray,
                       n_jobs: int = 0,
                       used_split: float = 1.,
                       verbosity: int = 5):
    """
    Compute the distance matrix for given samples
    :param all_samples: matrix of samples of shape (n_samples, n_dimensions),
        meaning every sample is a flat vector of n_dimensions
    :param n_jobs: number of cores in which this calculation should run
    :param used_split: what fraction of randomly selected
        samples should be used for the calculation.
        By default, use 1., if this takes too long, lower this number.
    :param verbosity: Verbosity to be passes to joblib.Parallel (0 for none, 10 for all)
    :return: distance matrix of shape (n_dimensions, n_dimensions)

    """
    return 1 - np.corrcoef(all_samples.T)


def distances(samples,
              verbose=False):
    """
    Compute the distance matrix for given samples
    :param samples: matrix of samples of shape (n_samples, n_dimensions),
        meaning every sample is a flat vector of n_dimensions
    :param verbose: should this print progress statements
    :return: distance matrix of shape (n_dimensions, n_dimensions)
    """

    def distance_inner(x, y):
        return np.sum(np.abs(x - y))

    ret: list[np.ndarray] = [np.array(0) for _ in range(samples.shape[1])]
    for i in range(samples.shape[1]):
        if verbose:
            print(f"Calculating dimension {i}...")
        ret[i] = np.array(
            [np.sum(list(map(distance_inner, samples[:, i], samples[:, j]))) for j in range((samples.shape[1]))])
    return np.array(ret)
