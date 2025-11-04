import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
from collections import Counter
from sklearn.cluster import HDBSCAN
from smlm_clust_analysis.internals.file_io import extract_xy


## Cluster analysis functions

def hdbscan(locs: 'np.ndarray[np.float64]', min_n: int) -> 'np.ndarray[np.float64]':

    """"
    HDBSCAN clustering of localization data. Returns the localisation data
    with the cluster assignments for each localisation.

    In: locs---localization table (numpy array)
    min_n---minimum number of points for cluster classification (int)

    Out: localization table---with cluster classification and probabilities.
    """

    # Instantiate and fit
    hdbscan = HDBSCAN(min_cluster_size=min_n).fit(locs[:, 2:4].reshape(-1, 2))

    # Cluster labels
    labels = hdbscan.labels_

    # Reassgine label for noise to avoid zero division problems
    labels[(labels == 0)] = -1

    cluster_probabilities = hdbscan.probabilities_

    all_data = np.concatenate((locs, labels[:, np.newaxis], cluster_probabilities[:, np.newaxis]), axis=1).reshape(-1, 13)

    return all_data

def denoise_data(dbscan_data: 'np.ndarray[np.float64]', min_n: int) -> 'np.ndarray[np.float64]':

    """
    Removes clusters below a minimum localizations threshold and noise.
    Returns localisation data with 'noise clusters' and small clusters
    discarded.

    In: dbscan_data---localization table with dbscan data (np array)
    min_n---minimum number of points for a cluster to be retained
    following filtering (int)
    """

    # Remove noise
    noiseless_data = dbscan_data[(dbscan_data[:, -2] > 0)]

    # Count how many localisations are assigned to each cluster
    label_count = dict(Counter(noiseless_data[:, -2]))

    for label, count in label_count.items():

        # If no. of localizations is below the threshold, discard from data
        if count < min_n:
                
            noiseless_data = noiseless_data[(noiseless_data[:, -2] != float(label))]
    
    return noiseless_data

def load_dbscan_data(path: str) -> 'np.ndarray[np.float64]':

    """
    This function loads the file containing the results of HDBSCAN.

    In: path---file path for HDBSCAN results (str)

    Out: data---HDBSCAN results (np array)
    """

    data = np.genfromtxt(path, dtype=float, delimiter=',',
                         skip_header=1)
    
    return data.reshape(-1, 13)

def calc_percent_coloc(dbscan_data: 'np.ndarray[np.float64]', threshold: float=0.4) -> float:

    """
    This function calculates the percentage of colocalised molecules for each channel. Colocalisation is 
    defined as molecules with a colocalisation score above a threshold. By default, this is set to 0.4

    In: dbscan_data---results from HDSCAN (np array)
    threshold---the number above which a molecule is considered to be colocalised (float)

    Out: coloc_percent_ch1---the percentage of molecules that are colocalised for the first channel (float)
    coloc_percent_ch2---the percentage of molecules that are colocalised for the second channel (float)
    """

    ch1 = dbscan_data[(dbscan_data[:, -4] == 1)]

    ch2 = dbscan_data[(dbscan_data[:, -4] == 2)]

    coloc_percent_ch1 = ch1[(ch1[:, -3] > threshold)].shape[0] * 100 / ch1.shape[0]

    coloc_percent_ch2 = ch2[(ch2[:, -3] > threshold)].shape[0] * 100 / ch2.shape[0]

    return coloc_percent_ch1, coloc_percent_ch2


def separate_coloc_data(dbscan_data: 'np.ndarray[np.float64]', threshold: float=0.4) -> 'np.ndarray[np.float64]':

    """
    This function separates the results from HDBSCAN into two separate arrays depending
    on whether the molecule colocalises or not. The threshold for colocalisation is 0.4

    In: dbscan_data---results from HDSCAN (np array)
    threshold---the number above which a molecule is considered to be colocalised (float)

    Out: no_coloc---localisations that were not colocalised (np array)
    coloc---localisations that were colocalised (np array)
    """

    no_coloc = dbscan_data[(dbscan_data[:, 10] < threshold)]

    coloc = dbscan_data[(dbscan_data[:, 10] > threshold)]

    return no_coloc, coloc

def calculate_ch1_intensity(cluster_points: 'np.ndarray[np.float64]') -> int:

    """
    Calculates cluster intensity, i.e. how many localisations of channel 1
    per cluster.

    In: cluster_points---cluster data (np array)

    Out: the number of xy localisations, indicator of cluster intensity.
    Not to be confused with fluorescent intensity
    """

    ch1_indices = np.where(cluster_points[:, 9] == 1)[0]

    if not ch1_indices.size > 0:
        return 0
    else:
        ch1_locs = cluster_points[ch1_indices, :]

        return 100 * np.size(ch1_locs, axis=0) / cluster_points.shape[0]

def calculate_center_of_mass(points: 'np.ndarray[np.float64]') -> 'np.ndarray[np.float64]':

    """
    Calculates cluster centroid.

    In: xy localisations (numpy array)

    Out: xy coordinate of the center of the cluster.
    """

    center = np.mean(points, axis=0)

    return center

def calculate_clust_area_perim(points: 'np.ndarray[np.float64]') -> float:

    """
    Calculates cluster area and perimeter with the Convexhull method.

    In: xy localisations (np array)

    Out: 'volume' enclosed by points. For 2D data, the volume corresponds
    to the area.
    'Area' enclosed by points. For 2D data, the area corresponds to the
    perimeter
    """

    return ConvexHull(points).volume, ConvexHull(points).area

def calculate_circularity(perimeter: float, area: float) -> float:

    """
    Calculates circularity by taking the ratio of the area to the perimeter.

    In: perimeter---cluster perimeter (float)
    area---cluster area (float)

    Out: circularity of cluster (float)
    """

    return 4 * np.pi * area / perimeter**2 

def calculate_radius(points: 'np.ndarray[np.float64]', center: 'np.ndarray[np.float64]') -> float:

    """
    Radius calculation. First calculates the pairwise distance of all
    cluster points from the centroid then selects the maximum.

    In: points---cluster xy localisations (np array)
    center---xy coordinates of cluster center

    Out: radius of cluster calculated by extracting the maximum of
    all pairwise distances.
    """
    
    return np.max(pairwise_distances(points, center))

def calculate_cluster_density(intensity: int, area: float) -> float:

    """
    This function calculates the density of a cluster. The density is
    defined as the number of molecules divided by the area of the cluster and
    has units of molecules/um^2. The area in nm^2 is first converted to um^2 for
    easier-to-read numbers.

    In: intensity---the number of molecules in the cluster (int)
    area---the cluster area (float)

    Out: cluster density---the number of molecules per square micrometer.
    """

    area_in_um = area / 1000000

    return intensity / area_in_um

def analyse_clusters(dbscan_data: 'np.ndarray[np.float64]') -> 'np.ndarray[np.float64]':

    """
    This function loops through each cluster label, extracts the xy localisations,
    then calculates cluster intensity, cluster density, area, and radius.

    In: dbscan_data---localization table with dbscan data (np array)

    Out: n x 8 table containing the centroid, cluster area, cluster radius,
    cluster circularity, cluster intensity, cluster denstiy, and cluster label (np array)
    """

    analysis_results = []

    cluster_labels = np.unique(dbscan_data[:, -2])

    for label in cluster_labels:

        cluster_points = dbscan_data[(dbscan_data[:, -2] == label)]

        cluster_points_xy = extract_xy(cluster_points)

        intensity = calculate_ch1_intensity(cluster_points)

        center_of_mass = calculate_center_of_mass(cluster_points_xy)

        center = center_of_mass[:, np.newaxis]

        cluster_area, cluster_perim = calculate_clust_area_perim(cluster_points_xy)

        cluster_radius = calculate_radius(cluster_points_xy, center.T)

        circularity = calculate_circularity(cluster_perim, cluster_area)

        cluster_density = calculate_cluster_density(cluster_points.shape[0], cluster_area)

        analysis_results.append([center_of_mass[0], center_of_mass[1], cluster_area,
                                 cluster_radius, circularity, intensity, cluster_density, label])
        
    return np.array(analysis_results).reshape(-1, 8)

def filter_clusters(cluster_data: 'np.ndarray[np.float64]') -> 'np.ndarray[np.float64]':

    """
    Remove clusters with very large radii or with radii below the limits
    of localization precision, and remove clusters with no locs from channel 1.
    Also ensures no values are nan.

    In: cluster_data---table with cluster statisitcs (np array)

    Out: filtered cluster data (np array)
    """
    cluster_data_nanfilt = cluster_data[~np.isnan(cluster_data).any(axis=1), :]

    filtered_clust_data = cluster_data_nanfilt[(cluster_data_nanfilt[:, 3] < 400)]

    filt_r = filtered_clust_data[(filtered_clust_data[:, 3] > 20)]

    filt_ch1_count = filt_r[(filt_r[:, 5] > 0)]

    filt_all = filt_ch1_count[(filt_ch1_count[:, 5] < 100)]

    return filt_all


def convert_to_dataframe(filt_cluster_data: 'np.ndarray[np.float64]') -> pd.DataFrame:

    """
    Converts cluster analysis results to a dataframe.

    In: filt_cluster_data---table containing cluster statisitcs filtered
    by radius size (np array).

    Out: the same but as a pandas dataframe.
    """

    cols = [
        'x (nm)',
        'y (nm)',
        'Area (nm^2)',
        'Radius (nm)',
        'Circularity',
        'Percent of Channel 1',
        'Density (n . um^-2)',
        'label',
    ]

    cluster_data_df = pd.DataFrame(data=filt_cluster_data, columns=cols)

    return cluster_data_df

def calculate_statistics(filt_cluster_data: pd.DataFrame) -> dict:

    """
    This function calculates the mean, median, and standard deviation
    for each cluster descriptior. That is the area, intensity, circularity,
    and radius.

    In: filt_cluster_data, table of cluster descriptors (np array)

    Out: clust_statistics. The mean, median, and standard deviation for
    each descriptor (dict).
    """

    clust_statistics = {}

    for i in range(2, len(filt_cluster_data.columns) - 1):

        statistics = {}

        statistics['Mean'] = np.mean(filt_cluster_data[filt_cluster_data.columns[i]])

        statistics['Median'] = np.median(filt_cluster_data[filt_cluster_data.columns[i]])

        statistics['Standard deviation'] = np.std(filt_cluster_data[filt_cluster_data.columns[i]])

        clust_statistics[filt_cluster_data.columns[i]] = statistics

    return clust_statistics
