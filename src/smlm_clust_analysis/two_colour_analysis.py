import numpy as np
from numba import jit, prange
import cv2 as cv
from scipy import stats
from scipy.spatial import cKDTree
import time
import file_io as io
from cluster_detection import generate_radii


def calculate_transformation_matrix(channel1: 'np.ndarray[np.float32]', channel2: 'np.ndarray[np.float32]') -> 'np.ndarray[np.float32]':

    """ Note: this function registers the first channel to the second channel
    I.e. it shifts the first channel to the second

    In: channel1---xy localisations from one channel (np array)
    channel2---xy localisations from the second channel (np array)

    Out: M---a 2x3 affine matrix that translates channel 1 to channel 2
    (np array)
    """

    M, _ = cv.estimateAffinePartial2D(channel1, channel2)

    return M

def register_channel(channel: 'np.ndarray[np.float32]', matrix: 'np.ndarray[np.float32]') -> 'np.ndarray[np.float32]':

    """
    This function takes a transformation matrix, applies the matrix
    to the localisations of one channel, thus correcting for chromatic
    aberrations. Usage note: if the order for calculating the matrix was
    green, red then input the green channel into this function. If it was
    the other way around then input the red channel.

    In: channel---the xy localisations of a particular channel (np array)
    matrix---the transformation matrix (2x3 np array)

    Out: corrected_channel---xy localisations of a particular channel following
    transformation/translation using the input matrix (np array)
    """

    # Use the first channel

    corrected_channel = cv.transform(np.array([channel]), matrix)

    return corrected_channel.reshape(channel.shape[0], 2)

def filter_bead_locs(ch1_locs: 'np.ndarray[np.float32]', ch2_locs: 'np.ndarray[np.float32]', nneighbors: 'np.ndarray[np.float32]') -> 'np.ndarray[np.float32]':

    """
    This function removes bead localisations following registration if the distance to its nearest neighbor
    is less than 50 nm.

    In: ch1_locs---xy localisations of one channel (np array)
    ch2_locs---xy localisations of second channel (np array)
    nneighbors---nearest neighbor for each localisation of channel 1 (np array)

    Out: ch1_filt---xy localisations of channel one following filtering.
    ch2_filt---xy localisations of channel two following filtering.
    """

    all_data = np.hstack((ch1_locs, ch2_locs, nneighbors)).reshape(-1, 5)

    filt_data = all_data[(all_data[:, 4] < 50)]

    ch1_filt, ch2_filt = filt_data[:, 0:2], filt_data[:, 2:4]
    
    return ch1_filt.astype(np.float32), ch2_filt.astype(np.float32)


## Functions for two-color STORM---CBC analysis

def add_channel(locs: 'np.ndarray[np.float32]', channel: int) -> 'np.ndarray[np.float32]':

    """

    This function adds a column to specify the appropriate channel
    to the localisation table.

    In: locs---localisation table (np array)
    channel---the wavelength of emission (int)

    Out: localisation table with additional column that specifies the 
    channel.
    """

    channel_col = np.repeat(channel, locs.shape[0]).reshape(locs.shape[0], 1)

    return np.hstack((locs, channel_col)).reshape(locs.shape[0], 10)

@jit(nopython=True, nogil=True, cache=False)
def calc_counts_with_radius(locs: 'np.ndarray[np.float32]', x0: float, y0: float, radii: list[float]) -> 'np.ndarray[np.int64]':

    """
    Calculate number of localisations from a list of increasing radii.

    In: locs---localisation table (np array)
    x0---x-coordinate of circle center (float or int)
    y0---y-coordinate of circle center (float or int)
    radii---increasing radii specifying a circle (list of float)

    Out: loc_counts_with_r---the number of localisations within circles
    of varying radii (np array of int)
    """

    loc_counts_with_r = np.zeros((1, len(radii)))

    for i in range(1, len(radii)):

        filt_locs_max_r = locs[((locs[:, 0] - x0)**2 + 
                                (locs[:, 1] - y0)**2 < (2 + max(radii))**2)]

        filt_locs = filt_locs_max_r[((filt_locs_max_r[:, 0] - x0)**2 + 
                                (filt_locs_max_r[:, 1] - y0)**2 < radii[i]**2)]
        
        filt_locs = filt_locs[((filt_locs[:, 0] - x0)**2 + 
                                (filt_locs[:, 1] - y0)**2 > radii[i-1]**2)]
        
        loc_counts_with_r[0, i] = filt_locs.shape[0] + 1
    
    return loc_counts_with_r.astype(np.float32)

def convert_radii_to_areas(radii: list[float]) -> list[float]:

    """
    This function converts the list of radii to a list of the differences
    between consecutive radii squared.

    In: radii---list of radii over which cbc will be calculated

    Out: areas---the differences between radii squared
    """
    
    radii = radii.copy()

    radii.insert(0, 0)

    areas = [(radii[i+1])**2 - (radii[i])**2 for i in range(0, len(radii) - 1)]

    return areas

@jit(nopython=True, nogil=True, cache=False)
def calc_loc_distribution(counts: 'np.ndarray[np.int64]', radii: list[float], areas: list[float]) -> 'np.ndarray[np.float32]':

    """
    Calculates distribution of number of localisations with increasing radii
    from a localisation.

    In: counts---the number of localisations at various radii (np array)
    radii---circle radii (list of floats)

    Out: cbc---coordinate-based colocalisation (np array)
    """

    max_r = max(radii)
    areas = np.array(areas).astype(np.float32)

    d = counts / np.sum(counts) * (max_r ** 2 / areas ** 2)

    return d

@jit(nopython=True, nogil=True, cache=False, parallel=True)
def calc_all_distributions(channel1_locs: 'np.ndarray[np.float32]', channel2_locs: 'np.ndarray[np.float32]', radii: list[float], areas: list[float]) -> 'np.ndarray[np.float32]':

    """
    Combines the previous two functions to calculate distributions along
    an increasing radius for all distributions of a particular channel.

    In: channel1_locs---localisation table for a particular channel (np array)
    channel2_locs---localisation table for second channel (np array)
    radii---incrementally increasing radii (list of floats)
    area---incrementally increasing areas (list of floats)


    Out: dist_ch1---distribution of cbc values for channel 1 relative to channel 2
    dist_ch2---distribution of cbc values for channel 2 relative to channel 1
    """

    dist_ch1 = np.zeros((channel1_locs.shape[0], len(areas)), dtype=np.float32)

    dist_ch2 = np.zeros((channel1_locs.shape[0], len(areas)), dtype=np.float32)

    # Loop through all localisations

    for i in prange(0, channel1_locs.shape[0]):

        x0, y0 = channel1_locs[i, 0], channel1_locs[i, 1]

        # Channel 1
        ch1_counts = calc_counts_with_radius(
            channel1_locs, x0, y0, radii
        )

        dist_ch1[i, :] = calc_loc_distribution(ch1_counts, radii, areas)

        # Channel 2
        ch2_counts = calc_counts_with_radius(
            channel2_locs, x0, y0, radii
        )

        dist_ch2[i, :] = calc_loc_distribution(ch2_counts, radii, areas)
    
    return dist_ch1, dist_ch2

def calc_spearman_cor_coeff(ch1_dist: 'np.ndarray[np.float32]', ch2_dist: 'np.ndarray[np.float32]') -> 'np.ndarray[np.float32]':

    """
    Calculate Spearman correlation coefficients on a row-by-row basis
    for the distributions of channel 1 to itself and to channel 2. I.e, 
    between D_AA(r) and D_AB(r).

    In: ch1_dist---the distribution of the number of localisations
    for all localisations of a particular channel wrt to itself
    (np array)
    ch2_dist---the distribution of the number of localisations for
    all localisations of a particular channel wrt to another channel
    (np array)

    Out: spearman_cor_coeffs---spearman correlation coefficients for 
    all localisations (np array)
    """

    spearman_cor_coeffs = np.zeros((ch1_dist.shape[0], 1), dtype=np.float32)

    for i in range(0, ch1_dist.shape[0]):

        rho = stats.spearmanr(ch1_dist[i, :], ch2_dist[i, :]).statistic

        spearman_cor_coeffs[i, 0] = rho

    return spearman_cor_coeffs.reshape(ch1_dist.shape[0], 1)

def calculate_nneighbor_dist(ch1_locs: 'np.ndarray[np.float32]', ch2_locs: 'np.ndarray[np.float32]', radii: list[float]) -> 'np.ndarray[np.float32]':

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

    tree = cKDTree(ch2_locs)
    distances, _ = tree.query(ch1_locs, k=1)  # k=1 for nearest neighbor
    
    return distances.reshape(-1, 1) / max(radii)

def calc_coloc_values(spearman: 'np.ndarray[np.float32]', ch1_locs: 'np.ndarray[np.float32]', ch2_locs: 'np.ndarray[np.float32]', radii: list[float]) -> 'np.ndarray[np.float32]':

    """
    This function weights the spearman correlation coefficients
    with an exponential function.

    In: spearman---spearman correlation coefficients (np array).
    ch1_locs---localisation table for a particular channel (np array)
    ch2_locs---localisation table for second channel (np array)
    radii---incrementally increasing radii (list of floats)

    Out: correlation coefficients weighted by distances (np array)
    """

    nearest_neighbors = calculate_nneighbor_dist(ch1_locs,
                                                 ch2_locs, radii)
    
    cbcs = spearman * np.exp(-nearest_neighbors)

    return cbcs.reshape(ch1_locs.shape[0], 1)

def add_coloc_values(locs: 'np.ndarray[np.float32]', coloc_values: 'np.ndarray[np.float32]') -> 'np.ndarray[np.float32]':

    """
    Add correlation coefficients to localisation data.

    In: locs---localisation table (np array)
    coloc_values---weighted colocalisation correlation coefficients (np array)

    Out: localisation table with correlation coefficients
    """

    return np.hstack((locs, coloc_values)).reshape(locs.shape[0], 11)


def combine_channel_locs(ch1_locs: 'np.ndarray[np.float32]', ch2_locs: 'np.ndarray[np.float32]'):

    """
    Combines the localisations of channel one, and two. Recommended to do
    this following colocalisation analysis.

    In: ch1_locs---localisations of channel 1 with correlation coefficients
    (np array)
    ch2_locs---localisations of channel 2 with correlation coefficients (np array)

    Out: localisations of all channels (np array)
    """

    return np.vstack((ch1_locs, ch2_locs))

def main():

    start = time.perf_counter()

    green_bead_ch_path = "C:/Users/mxq76232/Downloads/test_coloc/bead_locs_488_filt.csv"

    red_bead_ch_path = "C:/Users/mxq76232/Downloads/test_coloc/bead_locs_640.csv"

    green_ch_path = "C:/Users/mxq76232/Downloads/test_coloc/reconstruction_filt_488.csv"

    red_ch_path = "C:/Users/mxq76232/Downloads/test_coloc/reconstruction_filt_640.csv"

    out = "C:/Users/mxq76232/Downloads/test_coloc"

    green_beads, red_beads = io.load_locs(path=green_bead_ch_path), io.load_locs(path=red_bead_ch_path)

    green_locs, red_locs = io.load_locs(path=green_ch_path), io.load_locs(path=red_ch_path)

    green_locs_xy = io.extract_xy(locs=green_locs)
    
    green_bead_xy, red_bead_xy = io.extract_xy(locs=green_beads), io.extract_xy(locs=red_beads)

    matrix = calculate_transformation_matrix(channel1=green_bead_xy, channel2=red_bead_xy)

    green_bead_xy_reg = register_channel(channel=green_bead_xy, matrix=matrix)

    nearest_neighbors = calculate_nneighbor_dist(ch1_locs=green_bead_xy_reg, ch2_locs=red_bead_xy, radii=[1, 1])

    green_xy_filt, red_xy_filt = filter_bead_locs(ch1_locs=green_bead_xy, ch2_locs=red_bead_xy, nneighbors=nearest_neighbors)

    matrix_2 = calculate_transformation_matrix(channel1=green_xy_filt, channel2=red_xy_filt)

    green_xy_reg = register_channel(channel=green_locs_xy, matrix=matrix_2)

    green_locs_cor = io.save_corrected_channels(cor_locs=green_xy_reg, locs=green_locs, out=out)

    green, red = add_channel(locs=green_locs_cor, channel=1), add_channel(locs=red_locs, channel=2)

    green_xy, red_xy = io.extract_xy(green), io.extract_xy(red)
    
    radii = generate_radii(bounding_radius=125, increment=25)

    areas = convert_radii_to_areas(radii)

    gg_dist, gr_dist = calc_all_distributions(green_xy, red_xy, radii, areas)
    
    green_spearman = calc_spearman_cor_coeff(gg_dist, gr_dist)

    colocs = calc_coloc_values(green_spearman, green_xy, red_xy, radii)

    io.save_locs_colocs(add_coloc_values(locs=green, coloc_values=colocs),
                       channel=1, out=out)
    
    rr_dist, rg_dist = calc_all_distributions(red_xy, green_xy, radii, areas)
    
    red_spearman = calc_spearman_cor_coeff(rr_dist, rg_dist)

    colocs_red = calc_coloc_values(red_spearman, red_xy, green_xy, radii)
    
    io.save_locs_colocs(add_coloc_values(locs=red, coloc_values=colocs_red),
                     channel=2, out=out)
    
    all_locs = combine_channel_locs(add_coloc_values(locs=green, coloc_values=colocs),
                                    add_coloc_values(locs=red, coloc_values=colocs_red))

    io.save_locs_colocs(all_locs, channel=3, out=out)

    end = time.perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))

if __name__ == "__main__":
    main()