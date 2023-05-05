import numpy as np


def shuffle_columns(X, ii0, iif):
    """Shuffles columns of X in [ii0, iif).
    
    Parameters
    ----------
    X : numpy.ndarray
    ii0 : int
    iif : int

    Returns
    -------
    numpy.ndarray
        The array with the shuffled columns.
    """

    X2 = X.copy()
    for ii in range(ii0, iif):
        np.random.shuffle(X2[:, ii])
    return X2
