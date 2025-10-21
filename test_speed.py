import numpy as np
from numba import jit, prange
from scipy.spatial import cKDTree

@jit(nopython=True, nogil=True, cache=False, parallel=True)
def calculate_nneighbor_dist(ch1_locs: 'np.ndarray[np.float64]', ch2_locs: 'np.ndarray[np.float64]') -> 'np.ndarray[np.float64]':

    """
    Calculate the nearest neighbours for a particular localisation
    and find the shortest distance between a localisation of channel 1
    to a localisation of channel 2. Overall, this function 'weighs'
    the spearman correlation coefficients by the distances from one
    localisation to another belonging to another species.

    In: channel1_locs---localisation table for a particular channel (np array)
    channel2_locs---localisation table for second channel (np array)
    radii---incrementally increasing radii (list of floats)

    Out: distances weighted by maximum radius (np array)

    """

    distances = np.zeros((ch1_locs.shape[0], 1))

    for i in prange(0, ch1_locs.shape[0]):

        x0, y0 = ch1_locs[i, 0], ch1_locs[i, 1]

        distances[i, 0] = np.min(np.sqrt((ch2_locs[:, 0] - x0)**2 + 
                                  (ch2_locs[:, 1] - y0)**2))
    
    return distances

@jit(nopython=True, cache=False, nogil=True)
def euclid_dist(x_dist, y_dist):

    return np.sqrt(x_dist**2 + y_dist**2)

@jit(nopython=True, cache=False, nogil=True, parallel=True)
def nneighbor_sort(ch1_locs, ch2_locs):

    distances = np.zeros((ch1_locs.shape[0], 1))

    for i in prange(0, ch1_locs.shape[0]):

        x0, y0 = ch1_locs[i, 0], ch1_locs[i, 1]

        s = 0

        for j in range(0, ch2_locs.shape[0]):

            x1, y1 = ch2_locs[j, 0], ch2_locs[j, 1]

            xdist, ydist = x1 - x0, y1 - y0

            if np.abs(xdist) < s or np.abs(ydist) < s:

                pass

            else:

                s = euclid_dist(xdist, ydist)
        
        distances[i, 0] = s
    
    return distances

def nneighbor_sort_scipy(ch1_locs, ch2_locs):
    tree = cKDTree(ch2_locs)
    distances, _ = tree.query(ch1_locs, k=1)  # k=1 for nearest neighbor
    return distances.reshape(-1, 1)

def test1():

    ch1 = np.random.rand(100000, 2)
    ch2 = np.random.rand(100000, 2)

    x = calculate_nneighbor_dist(ch1, ch2)

    return x

def test2():

    ch1 = np.random.rand(100000, 2)
    ch2 = np.random.rand(100000, 2)

    x = nneighbor_sort(ch1, ch2)

    return x

def test3():

    ch1 = np.random.rand(100000, 2)
    ch2 = np.random.rand(100000, 2)

    x = nneighbor_sort_scipy(ch1, ch2)

    return x


if __name__ == "__main__":

    import timeit
    print(timeit.timeit("test1()", number=100, setup="from __main__ import test1"))