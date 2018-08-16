import numpy as np

def rmseff(x, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    x = np.sort(x, kind='mergesort')
    m = int(c*len(x)) + 1
    return np.min(x_sorted[m:] - x_sorted[:-m])/2.0
