import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from collections import Counter
from sklearn.cluster import HDBSCAN
from scipy import stats
import ripleyk
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
import cv2 as cv

def user_input():

    """
    Accepts user input.

    In: None
    
    Out: input_text---string entered by the user.
    """

    counter = 0

    input_text = ''

    while True:

        if counter == 3:

            raise SystemError('Too many incorrect attempts, exiting.')
        
        input_text = input('')
        
        if len(input_text) > 0:

            break

        else:

            counter += 1

            print('Invalid input, you have 3 - ' + str(counter) + 'attempts remaining')
        
    return input_text

def load_locs(path: str, channels=1):

    """
    Extract localisation data from .csv file, preferably from ThunderSTORM
    
    In: path (str)---file path to localization data

    Out: locs---localization data from ThunderSTORM (numpy array)
    
    """

    locs = np.genfromtxt(path, delimiter=',', dtype=float,
                         skip_header=1)
    
    if channels == 1:
    
        return locs.reshape(-1, 9)
    
    else:

        return locs.reshape(-1, 11)

def extract_xy(locs):

    """
    Extract xy localisations.

    In: localization table (numpy array)

    Out: xy localizations (numpy array)
    """

    return locs[:, 2:4].reshape(-1, 2)

def generate_radii(bounding_radius, increment):

    """
    Generate a list of radii for cluster detection via Ripley functions.
    Bounding radius is the maximum radius while the increment defines the range 
    of radii for the functions.

    In: the maximum radius of a region of interest (float).
    The size of the increment, starting from r = 0 (float).

    Out: a list of numbers representing the increasing radii of a circle
    from a point (list of floats.)
    """

    radii = list(np.arange(0, bounding_radius, increment))

    return radii[1:]

def ripley_k_function(xy_data, r, br):

    """
    2D Ripley's K=function. Converts result to numpy array.

    In: xy_data---xy localisations from STORM (np array)
    r---list of radii to calculate ripley's K-values (list of floats)
    br---bounding radius, maximum radius to calc K-values (float)

    Out: array of K-values.
    """

    k = ripleyk.calculate_ripley(r, br, d1=xy_data[:, 0], d2=xy_data[:, 1])
    
    return np.array(k).reshape(len(k), 1)

def ripley_l_function(k_values):

    """
    2D Ripley's L-function, normalized such that the expected value is r.

    In: k_values---Ripley K-values (numpy array)

    Out: Ripley L-values (numpy array)

    """

    return np.sqrt(k_values / np.pi)

def ripley_h_function(l_values, radii):

    """
    2D Ripley's H-function, normalized such that the expected value is 0.
    The radii are converted to a numpy array.

    In: l_values---Ripley L-values (numpy array)

    Out: Ripley H-values (numpy array)
    """

    return l_values - np.array(radii).reshape(len(radii), 1)

def plot_ripley_h(h_values, radii, out):

    """
    Plots Ripley's H-function against radii.

    In: h_values---Ripley h-values (numpy array)
    radii---list of radii over which Ripley's K-function was calculated (np array)
    out---folder path where plot will be saved (str)
    title---the file name of the plot (str)

    Out: saves .png of H-values against radii.
    """

    plt.ioff()

    # Set font type and size for axis values
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 20

    fig, ax = plt.subplots(figsize=(7, 7), dpi=500)

    # Plot Ripley's H-function against radii
    ax.plot(radii, h_values, 'b', linewidth=5.0)
    ax.axhline(y=0, color='r')

    # Set axis limits
    ax.set_xlim(left=0)

    ratio = 1.0

    # Make sure figure is square
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    # Major and minor tick parameters
    ax.tick_params(axis='y', which='major', length=10, direction='in')
    ax.tick_params(axis='y', which='minor', length=5, direction='in')
    ax.tick_params(axis='x', which='major', length=10, direction='in')
    ax.tick_params(axis='x', which='minor', length=5, direction='in')

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    # Set colors of axes and box
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    # Axis labels
    ax.set_xlabel('Radius (nm)', labelpad=6, fontsize=30)
    ax.set_ylabel('H(r)', labelpad=2, fontsize=30)

    plt.savefig(out + '/ripley_hfunction.png')

def calculate_rmax(h_values, radii):

    """
    Calculate the radius at which Ripley's H-function is at a maximum

    In: h_values---Ripley h-values (numpy array)
    radii---list of radii over which Ripley's K-function was calculated (np array)

    Out: radius at which the H-values are at a maximum (float)
    """

    return radii[h_values.argmax()]

def save_max_r(outpath, max_r):

    """
    Save the value of the radius at which Ripley's H-function is at a maximum
    """

    with open(outpath + '/max_r.txt', 'w') as f:

        f.write('The maximum value of r is: ' + str(max_r)
                + ' nm')

def hdbscan(locs, min_n):

    """"
    HDBSCAN clustering of localization data. Returns the localisation data
    with the cluster assignments for each localisation.

    In: locs---localization table (numpy array)
    min_n---minimum number of points for cluster classification (int)

    Out: localization table---with cluster classification and probabilities.
    """

    # Instantiate and fit
    hdbscan = HDBSCAN(min_cluster_size=min_n).fit(locs[:, 2:4].reshape(-1, 2))

    # DBSCAN labels
    labels = hdbscan.labels_

    # Reassgine label for noise
    labels[(labels == 0)] = -1

    cluster_probabilities = hdbscan.probabilities_

    all_data = np.concatenate((locs, labels[:, np.newaxis], cluster_probabilities[:, np.newaxis]), axis=1).reshape(-1, 13)

    # Combine localisation data, labels, and probabilities
    return all_data

def denoise_data(dbscan_data, min_n):

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

def save_dbscan_results(data, n_channels, outpath, filt=0):

    """
    Convert dbscan results to dataframe and save as .csv. Columns change
    depending on whether data include localizations from one channel or 
    two channels. 

    In: data---localizations with DBSCAN cluster classification.
    n_channels---the number of channels. One for single-color STORM data,
    two for two-color STORM, etc. (int)
    outpath---folder where results will be saved (str).
    filt---whether the data has been filtered or not. 0 for no filtering,
    1 for filtered (int).
    """

    if n_channels == 1:
    
        cols = ['id',
            'frame',
            'x [nm]',
            'y [nm]',
            'sigma [nm]',
            'intensity [photons]',
            'offset [photons]',
            'bkgstd [photons]',
            'uncertainty [nm]',
            'label',
            'probability']
    
    else:

        cols = ['id',
            'frame',
            'x [nm]',
            'y [nm]',
            'sigma [nm]',
            'intensity [photons]',
            'offset [photons]',
            'bkgstd [photons]',
            'uncertainty [nm]',
            'Channel',
            'Degree of colocalisation',
            'label',
            'probability']
    
    dbscan_results_df = pd.DataFrame(data=data, columns=cols)

    if filt == 0:
    
        dbscan_results_df.to_csv(outpath + '/dbscan_output.csv', index=False)

    else:

        dbscan_results_df.to_csv(outpath + '/filt_dbscan_output.csv', index=False)

## Cluster analysis functions

def load_dbscan_data(path):

    """
    This function loads the file containing the results of HDBSCAN.

    In: path---file path for HDBSCAN results (str)

    Out: data---HDBSCAN results (np array)
    """

    data = np.genfromtxt(path, dtype=float, delimiter=',',
                         skip_header=1)
    
    return data.reshape(-1, 13)

def calc_percent_coloc(dbscan_data, threshold=0.4):

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

def plot_bar_percent_coloc(percent_ch1, percent_ch2, name_ch1, name_ch2, out):

    """"
    This function plots the percentage of colocalised molecules for each channel as a bar plot.

    In: percent_ch1---percentage of colocalised molecules for channel 1 (float)
    percent_ch2---same as above but for channel 2 (float)
    name_ch1---the name of channel 1, typically its colour (str)
    name_ch2---same but for channel 2
    out---path where bar plot will be saved.

    Out: None but a .png file will be saved in the specified folder.
    """

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 20

    fig, ax = plt.subplots(figsize=(7, 7), dpi=500)

    ax.bar([name_ch1, name_ch2], [percent_ch1, percent_ch2])

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')

    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_ylabel('Colocalised molecules (%)', labelpad=12, fontsize=28)

    plt.savefig(out + '/coloc_percent.png')


def separate_coloc_data(dbscan_data, threshold=0.4):

    """
    This function separates the results from HDBSCAN into two separate arrays depending
    on whether the molecule colocalises or not. The threshold for colocalisation is 0.4

    In: dbscan_data---results from HDSCAN (np array)
    threshold---the number above which a molecule is considered to be colocalised (float)

    Out: no_coloc---localisations that were not colocalised (np array)
    coloc---localisations that were colocalised (np array)
    """

    no_coloc = dbscan_data[(dbscan_data[:, -3] < threshold)]

    coloc = dbscan_data[(dbscan_data[:, -3] > threshold)]

    return no_coloc.reshape(-1, 13), coloc.reshape(-1, 13)

def calculate_intensity(points):

    """
    Calculates cluster intensity, i.e. how many localisations per cluster.

    In: points---xy localisations (np array)

    Out: the number of xy localisations, indicator of cluster intensity.
    Not to be confused with fluorescent intensity
    """

    return np.size(points, axis=0)

def calculate_center_of_mass(points):

    """
    Calculates cluster centroid.

    In: xy localisations (numpy array)

    Out: xy coordinate of the center of the cluster.
    """

    center = np.mean(points, axis=0)

    return center

def calculate_clust_area_perim(points):

    """
    Calculates cluster area and perimeter with the Convexhull method.

    In: xy localisations (np array)

    Out: 'volume' enclosed by points. For 2D data, the volume corresponds
    to the area.
    'Area' enclosed by points. For 2D data, the area corresponds to the
    perimeter
    """

    return ConvexHull(points).volume, ConvexHull(points).area

def calculate_circularity(perimeter, area):

    """
    Calculates circularity by taking the ratio of the area to the perimeter.

    In: perimeter---cluster perimeter (float)
    area---cluster area (float)

    Out: circularity of cluster (float)
    """

    return 4 * np.pi * area / perimeter**2 

def calculate_radius(points, center):

    """
    Radius calculation. First calculates the pairwise distance of all
    cluster points from the centroid then selects the maximum.

    In: points---cluster xy localisations (np array)
    center---xy coordinates of cluster center

    Out: radius of cluster calculated by extracting the maximum of
    all pairwise distances.
    """
    
    return np.max(pairwise_distances(points, center))

def analyse_clusters(dbscan_data):

    """
    This function loops through each cluster label, extracts the xy localisations,
    then calculates cluster intensity, area, and radius.

    In: dbscan_data---localization table with dbscan data (np array)

    Out: n x 7 table containing the centroid, cluster area, cluster radius,
    cluster circularity, cluster intensity, and cluster label (np array)
    """

    analysis_results = []

    cluster_labels = np.unique(dbscan_data[:, -2])

    for label in cluster_labels:

        cluster_points = dbscan_data[(dbscan_data[:, -2] == label)]

        cluster_points_xy = extract_xy(cluster_points)

        intensity = calculate_intensity(cluster_points_xy)

        center_of_mass = calculate_center_of_mass(cluster_points_xy)

        center = center_of_mass[:, np.newaxis]

        cluster_area, cluster_perim = calculate_clust_area_perim(cluster_points_xy)

        cluster_radius = calculate_radius(cluster_points_xy, center=center.T)

        circularity = calculate_circularity(perimeter=cluster_perim, area=cluster_area)

        analysis_results.append([center_of_mass[0], center_of_mass[1], cluster_area,
                                 cluster_radius, circularity, intensity, label])
        
    return np.array(analysis_results).reshape(-1, 7)

def filter_clusters(cluster_data):

    """
    Remove clusters with very large radii or very high
    intensities and  ensure no values are nan.

    In: cluster_data---table with cluster statisitcs (np array)

    Out: filtered cluster data (np array)
    """
    cluster_data_nanfilt = cluster_data[~np.isnan(cluster_data).any(axis=1), :]

    filtered_clust_data = cluster_data_nanfilt[(cluster_data_nanfilt[:, 3] < 400)]

    filt_r = filtered_clust_data[(filtered_clust_data[:, 3] > 20)]

    filt_all = filt_r[(filt_r[:, 5] < 100)]

    return filt_all


def convert_to_dataframe(filt_cluster_data):

    """
    Converts cluster analysis results to a dataframe.

    In: filt_cluster_data---table containing cluster statisitcs filtered
    by radius size (np array).

    Out: the same but as a pandas dataframe.
    """

    cols = [
        'x[nm]',
        'y[nm]',
        'Area [nm^2]',
        'Radius [nm]',
        'Circularity',
        'Intensity',
        'label'
    ]

    cluster_data_df = pd.DataFrame(data=filt_cluster_data, columns=cols)

    return cluster_data_df
    
def save_cluster_analysis(filt_cluster_data, outpath, coloc=0):

    """
    Save results of cluster analysis as a .csv file.

    In: filt_cluster_data---cluster statistics following filtering (np array)
    outpath---folder where data will be saved (str)
    coloc---specifies whether all data (coloc=0), non-colocalised data (coloc=1),
    or colocalised data (coloc=2) is to be saved (int)

    Out: None but a .csv file with cluster stats is saved.
    """

    if coloc == 0:

        filt_cluster_data.to_csv(outpath + '/cluster_analysis.csv', sep=',',
                                 index=False)

    elif coloc == 1:

        filt_cluster_data.to_csv(outpath + '/cluster_analysis_coloc_ch1.csv', sep=',',
                                 index=False)
    
    else:

        filt_cluster_data.to_csv(outpath + '/cluster_analysis_coloc_ch2.csv', sep=',',
                                 index=False)

def plot_histogram(data, title, out, coloc=0):

    """
    Plots a histogram for a column of data.

    In: data---column of a data describing a phenomenon (np array).
    title---name to give the plot (str)
    out---folder where plot will be saved (str)
    coloc---specifies whether all data (coloc=0), non-colocalised data (coloc=1),
    or colocalised data (coloc=2) is to be saved (int)

    Out: no output but image of plot will be saved in the specified folder.
    """

    plt.ioff()

    weights = np.ones_like(data) / float(len(data))

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 20

    fig, ax = plt.subplots(figsize=(7, 7), dpi=500)

    plt.hist(data, bins=20, weights=weights, edgecolor='black', linewidth=1.1, color='C3')

    ax.set_xlim(right=np.max(data) + 0.05 * np.max(data))
    
    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=10, direction='in')
    ax.tick_params(axis='y', which='minor', length=5, direction='in')
    ax.tick_params(axis='x', which='major', length=10, direction='in')
    ax.tick_params(axis='x', which='minor', length=5, direction='in')

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_xlabel(title, labelpad=6, fontsize=30)
    ax.set_ylabel('Frequency', labelpad=2, fontsize=30)

    if coloc == 0:
    
        plt.savefig(out + '/' + title + '.png')
    
    elif coloc == 1:

        plt.savefig(out + '/' + title + '_ch1_coloc.png')
    
    else:

        plt.savefig(out + '/' + title + '_ch2_coloc.png')

def plot_cluster_statistics(filt_cluster_data, outpath, coloc=0):

    """
    Plots and saves histograms for cluster intensity, area, and radius.

    In: filt_cluster_data---table with statistics for all clusters (np array)
    outpath---path to folder where plots will be saved (str)
    coloc---specifies whether all data (coloc=0), non-colocalised data (coloc=1),
    or colocalised data (coloc=2) is to be saved (int)

    Out: None but four plots should be saved in the specified folder.
    """

    for i in range(2, filt_cluster_data.shape[1] - 1):

        plot_histogram(filt_cluster_data[filt_cluster_data.columns[i]],
                      filt_cluster_data.columns[i], out=outpath,
                      coloc=coloc)

def calculate_statisitcs(filt_cluster_data):

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

def save_statistics(cluster_statistics, out, coloc=0):

    """
    This function saves the statistics as a .txt file.

    In: cluster_statistics. The mean, median, and standard deviation of each
    cluster descriptor (dict).
    out: output folder where the file will be saved (str).
    coloc---specifies whether all data (coloc=0), non-colocalised data (coloc=1),
    or colocalised data (coloc=2) is to be saved (int)

    Out: None but a .txt file should be saved in the specified output path.
    """

    if coloc == 0:

        with open(out + '/cluster_stats_summary.txt', 'w') as f:
            
            for stat in cluster_statistics:

                print(stat + ' ' + str(cluster_statistics[stat]) + '\n',
                    file=f)
    
    elif coloc == 1:

        with open(out + '/cluster_stats_summary_ch1_coloc.txt', 'w') as f:
            
            for stat in cluster_statistics:

                print(stat + ' ' + str(cluster_statistics[stat]) + '\n',
                    file=f)
    
    else:

        with open(out + '/cluster_stats_summary_ch2_coloc.txt', 'w') as f:
            
            for stat in cluster_statistics:

                print(stat + ' ' + str(cluster_statistics[stat]) + '\n',
                    file=f)

def plot_boxplot(data, statistic, out):

    """
    This function plots a boxplot.

    In:
    data---the data for the boxplot. Ideally should be a list of arrays
    statistic---the parameter that will be compared, i.e. the parameter on the y-axis (str)
    out---the output folder where the plot will be saved (str)

    Out:
    None but a .png file of the boxplot will be saved in the specified folder.
    """

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 20

    fig, ax = plt.subplots(figsize=(8, 8), dpi=500)

    ax.boxplot(data)

    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')

    ax.set_xticklabels(['Non-colocalised', 'Colocalised'], labelpad=4, fontsize=16)

    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_ylabel(statistic, labelpad=12, fontsize=28)

    plt.savefig(out + '/' + statistic + '.png')

def compare_clust_size(data, coloc_data, out):

    """
    This function extracts the cluster radii from colocalised molecules and non-colocalised
    molecules and plots it as a boxplot.

    In: data---cluster data from non-colocalised molecules (np array)
    coloc_data---cluster data from colocalised molecules (np array)
    out---output folder where things will be saved (str)

    Out: None but a boxplot comparing cluster radii will be saved to the specified folder.
    """

    no_loc_radii, loc_radii = data[:, 3], coloc_data[:, 3]

    radii_data = [no_loc_radii, loc_radii]

    plot_boxplot(radii_data, statistic='Radius (nm)', out=out)

def compare_clust_circularity(data, coloc_data, out):

    """
    This function extracts the cluster circularity from colocalised molecules and non-colocalised
    molecules and plots it as a boxplot.

    In: data---cluster data from non-colocalised molecules (np array)
    coloc_data---cluster data from colocalised molecules (np array)
    out---output folder where things will be saved (str)

    Out: None but a boxplot comparing cluster circularity will be saved to the specified folder.
    """

    no_loc_radii, loc_radii = data[:, 4], coloc_data[:, 4]

    radii_data = [no_loc_radii, loc_radii]

    plot_boxplot(radii_data, statistic='Circularity', out=out)

## Cluster visualisation

def make_circles(x, y, r):

    circles = [plt.Circle((xi, yi), radius=c, linewidth=0)
               for xi, yi, c in zip(x, y, r)]
    
    return circles

def extract_xyr(cluster_data):

    x = cluster_data[cluster_data.columns[0]]

    y = cluster_data[cluster_data.columns[1]]

    r = cluster_data[cluster_data.columns[3]]

    return x, y, r

def scale(data, scaling_factor):

    return data / scaling_factor

def plot_clusters(cluster_data, loc_data, out, title):

    x, y, r = extract_xyr(cluster_data=cluster_data)

    x, y, r = scale(x, scaling_factor=19), scale(y, scaling_factor=19), scale(r, scaling_factor=19)

    xy_locs = scale(extract_xy(locs=loc_data))
    
    px = 1/plt.rcParams['figure.dpi']

    fig, ax = plt.subplot(figsize=(2000*px, 2000*px), dpi=500)

    clusters = mpl.collections.PatchCollection(
        make_circles(x, y, r), facecolor='none', color='b')

    ax.scatter(xy_locs[:, 0], xy_locs[:, 1], s=20,
               facecolors='r', edgecolors='r')
    
    ratio = 1.0
    
    ax.set_xlim([0, 2100])
    ax.set_ylim([0, 2100])

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.add_artist(clusters)

    ax.tick_params(axis='y', which='major', length=10, direction='in')
    ax.tick_params(axis='y', which='minor', length=5, direction='in')
    ax.tick_params(axis='x', which='major', length=10, direction='in')
    ax.tick_params(axis='x', which='minor', length=5, direction='in')

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    ax.set_xlabel('x (nm)', labelpad=6, fontsize=40)
    ax.set_ylabel('y (nm)', labelpad=2, fontsize=40)

    plt.savefig(out + '/' + title + '.png')

## Functions for two-color STORM

def extract_xy_cr(locs):

    """
    Extract xy locs again but as 32 bit floating integers.

    In: locs---localisation table (np array)

    Out: xy localisations (np array, 32-bit floats)
    """

    return locs[:, 2:4].reshape(-1, 2).astype(np.float32)

def calculate_transformation_matrix(channel1, channel2):

    """ Note: this function registers the first channel to the second channel
    I.e. it shifts the first channel to the second

    In: channel1---xy localisations from one channel (np array)
    channel2---xy localisations from the second channel (np array)

    Out: M---a 2x3 affine matrix that translates channel 1 to channel 2
    (np array)
    """

    M, inliers = cv.estimateAffinePartial2D(channel1, channel2)

    return M

def register_channel(channel, matrix):

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

def measure_accuracy(bead1_loc_reg, bead2_loc):

    n_neighbors = np.min(pairwise_distances(bead1_loc_reg, bead2_loc))

    return n_neighbors.reshape(bead1_loc_reg.shape[0], 1)


def compare_channels(channel1, channel2):

    """
    This function plots the xy-localisations of channel 1 and channel 2
    as a scatter plot and saves the plot in a specified folder.

    In: channel1---xy localisations of one channel (np array)
    channel2---xy localisations of second channel (np array)
    out---output folder where plot will be saved (str)

    Out: None but the plot should be saved as a .png file in the 
    output folder.
    """

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 10

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.scatter(channel1[:, 0], channel1[:, 1], s=2,
               facecolors='b')
    ax.scatter(channel2[:, 0], channel2[:, 1], s=2,
               facecolors='r')
    
    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=5, direction='out')
    ax.tick_params(axis='y', which='minor', length=2, direction='out')
    ax.tick_params(axis='x', which='major', length=5, direction='out')
    ax.tick_params(axis='x', which='minor', length=2, direction='out')

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)

    plt.show()

def save_corrected_channels(cor_locs, locs, out):

    """
    This function combines the registered xy localisations with the
    rest of the localisation table and saves it as a .csv file. Note:
    this function probably needs to be refactored to save it as a dataframe.

    In: cor_locs---registered xy localisations (np array)
    locs---the full localisation table (np array)
    out---user-specified output folder (str)

    Out: cor_data---localisation table with registered xy localisations.
    A .csv file is also saved in the specified output folder.
    """

    cor_data = np.hstack((locs[:, 0:2], cor_locs, locs[:, 4:])).reshape(locs.shape[0], 9)

    cols = ['id',
            'frame',
            'x [nm]',
            'y [nm]',
            'sigma [nm]',
            'intensity [photons]',
            'offset [photons]',
            'bkgstd [photons]',
            'uncertainty [nm]'
            ]

    locs_df = pd.DataFrame(data=cor_data, columns=cols)

    locs_df.to_csv(out + '/corrected_locs.csv', index=False)
    
    return cor_data

## Functions for two-color STORM---CBC analysis

def add_channel(locs, channel: int):

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
def calc_counts_with_radius(locs, x0, y0, radii):

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

        filt_locs = locs[((locs[:, 0] - x0)**2 + 
                                (locs[:, 1] - y0)**2 < radii[i]**2)]
        
        filt_locs = filt_locs[((filt_locs[:, 0] - x0)**2 + 
                                (filt_locs[:, 1] - y0)**2 > radii[i-1]**2)]
        
        loc_counts_with_r[0, i] = filt_locs.shape[0] + 1
    
    return loc_counts_with_r

@jit(nopython=True, nogil=True, cache=False)
def calc_loc_distribution(counts, radii):

    """
    Calculates distribution of number of localisations with increasing radii
    from a localisation.

    In: counts---the number of localisations at various radii (np array)
    radii---circle radii (list of floats)

    Out: cbc---coordinate-based colocalisation (np array)
    """

    max_r = max(radii)

    cbc = counts / np.sum(counts) * (max_r ** 2 / np.array(radii) ** 2)

    return cbc

@jit(nopython=True, nogil=True, cache=False)
def calc_all_distributions(channel1_locs, channel2_locs, radii):

    """
    Combines the previous two functions to calculate distributions along
    an increasing radius for all distributions of a particular channel.

    In: channel1_locs---localisation table for a particular channel (np array)
    channel2_locs---localisation table for second channel (np array)
    radii---incrementally increasing radii (list of floats)

    Out: dist_ch1---distribution of cbc values for channel 1 relative to channel 2
    dist_ch2---distribution of cbc values for channel 2 relative to channel 1
    """

    dist_ch1 = np.zeros((channel1_locs.shape[0], len(radii)))

    dist_ch2 = np.zeros((channel1_locs.shape[0], len(radii)))

    # Loop through all localisations

    for i in range(0, channel1_locs.shape[0]):

        x0, y0 = channel1_locs[i, 0], channel1_locs[i, 1]

        # Channel 1
        ch1_counts = calc_counts_with_radius(
            locs=channel1_locs, x0=x0, y0=y0, radii=radii
        )

        dist_ch1 = calc_loc_distribution(counts=ch1_counts, radii=radii)

        # Channel 2
        ch2_counts = calc_counts_with_radius(
            locs=channel2_locs, x0=x0, y0=y0, radii=radii
        )

        dist_ch2 = calc_loc_distribution(counts=ch2_counts, radii=radii)
    
    return dist_ch1, dist_ch2

def calc_spearman_cor_coeff(ch1_dist, ch2_dist):

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

    spearman_cor_coeffs = np.zeros((ch1_dist.shape[0], 1))

    for i in range(0, ch1_dist.shape[0]):

        rho = stats.spearmanr(ch1_dist[i, :], ch2_dist[i, :]).statistic

        spearman_cor_coeffs[i, 0] = rho

    return spearman_cor_coeffs.reshape(ch1_dist.shape[0], 1)

@jit(nopython=True, nogil=True, cache=False)
def calculate_nneighbor_dist(ch1_locs, ch2_locs, radii):

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

    for i in range(0, ch1_locs.shape[0]):

        x0, y0 = ch1_locs[i, 0], ch1_locs[i, 1]

        distances[i, 0] = np.min(np.sqrt((ch2_locs[:, 0] - x0)**2 + 
                                  (ch2_locs[:, 1] - y0)**2))
    
    return distances / max(radii)

@jit(nopython=True, nogil=True, cache=False)
def calc_coloc_values(spearman, ch1_locs, ch2_locs, radii):

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

def add_coloc_values(locs, coloc_values):

    """
    Add correlation coefficients to localisation data.

    In: locs---localisation table (np array)
    coloc_values---weighted colocalisation correlation coefficients (np array)

    Out: localisation table with correlation coefficients
    """

    return np.hstack((locs, coloc_values)).reshape(locs.shape[0], 11)

def save_locs_colocs(processed_data, channel, out):

    """
    Save localisations with correlation coefficients as .csv file. Note
    this should probably be refactored to save data as a pd dataframe.

    In: processed_data---localisation table with correlation coefficients 
    (np array)
    out---user-specified output folder (str)

    Out: None but a .csv file should be saved.
    """

    cols = ['id',
            'frame',
            'x [nm]',
            'y [nm]',
            'sigma [nm]',
            'intensity [photons]',
            'offset [photons]',
            'bkgstd [photons]',
            'uncertainty [nm]',
            'channel',
            'cor_coeff'
            ]

    locs_proc = pd.DataFrame(data=processed_data, columns=cols)

    locs_proc.to_csv(out + '/processed_locs_' +
    str(channel) + '.csv', index=False)

def combine_channel_locs(ch1_locs, ch2_locs):

    """
    Combines the localisations of channel one, and two. Recommended to do
    this following colocalisation analysis.

    In: ch1_locs---localisations of channel 1 with correlation coefficients
    (np array)
    ch2_locs---localisations of channel 2 with correlation coefficients (np array)

    Out: localisations of all channels (np array)
    """

    return np.vstack((ch1_locs, ch2_locs))

## Main functions

def test_ripley_clustering():

    print('Enter path to localisation file')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()
    
    data = load_locs(path)

    xy = extract_xy(data)
    
    radii = generate_radii(bounding_radius=1500,
                           increment=10)
    
    k_values = ripley_k_function(xy, r=radii, br=1500)

    l_values = ripley_l_function(k_values=k_values)

    h_values = ripley_h_function(l_values=l_values, radii=radii)

    plot_ripley_h(h_values=h_values, radii=radii, out=outpath)

    rmax = calculate_rmax(h_values=h_values, radii=radii)

    save_max_r(outpath=outpath, max_r=rmax)

def two_color_reg_accuracy():

    print('Enter path to beads for green channel.')
    green_bead_ch_path = user_input()

    print('Enter path to beads for red channel.')
    red_bead_ch_path = user_input()

    print('Where you want things saved.')
    out = user_input()

    green_beads, red_beads = load_locs(path=green_bead_ch_path), load_locs(path=red_bead_ch_path)

    green_bead_xy, red_bead_xy = extract_xy_cr(locs=green_beads), extract_xy_cr(locs=red_beads)

    matrix = calculate_transformation_matrix(channel1=green_bead_xy, channel2=red_bead_xy)

    green_xy_reg = register_channel(channel=green_bead_xy, matrix=matrix)

    nearest_neighbors = measure_accuracy(bead1_loc_reg=green_xy_reg, bead2_loc=red_bead_xy)

    plot_histogram(data=nearest_neighbors, title='reg_accuracy', out=out)

def two_color_analysis_all():

    print('Enter path to beads for green channel.')
    green_bead_ch_path = user_input()

    print('Enter path to beads for red channel.')
    red_bead_ch_path = user_input()

    print('Enter path to localisations in green channel.')
    green_ch_path = user_input()

    print('Enter path to localisations in red channel.')
    red_ch_path = user_input()

    print('Where you want things saved.')
    out = user_input()

    green_beads, red_beads = load_locs(path=green_bead_ch_path), load_locs(path=red_bead_ch_path)

    green_locs, red_locs = load_locs(path=green_ch_path), load_locs(path=red_ch_path)

    green_locs_xy = extract_xy_cr(locs=green_locs)
    
    green_bead_xy, red_bead_xy = extract_xy_cr(locs=green_beads), extract_xy_cr(locs=red_beads)

    matrix = calculate_transformation_matrix(channel1=green_bead_xy, channel2=red_bead_xy)

    green_xy_reg = register_channel(channel=green_locs_xy, matrix=matrix)

    green_locs_cor = save_corrected_channels(cor_locs=green_xy_reg, locs=green_locs, out=out)

    green, red = add_channel(locs=green_locs_cor, channel=1), add_channel(locs=red_locs, channel=2)

    green_xy, red_xy = extract_xy(green), extract_xy(red)
    
    radii = generate_radii(bounding_radius=200, increment=10)

    gg_dist, gr_dist = calc_all_distributions(channel1_locs=green_xy,
                                              channel2_locs=red_xy,
                                              radii=radii)
    
    green_spearman = calc_spearman_cor_coeff(ch1_dist=gg_dist, ch2_dist=gr_dist)

    colocs = calc_coloc_values(spearman=green_spearman, ch1_locs=green_xy,
                               ch2_locs=red_xy, radii=radii)

    save_locs_colocs(add_coloc_values(locs=green, coloc_values=colocs),
                       channel=1, out=out)
    
    rr_dist, rg_dist = calc_all_distributions(channel1_locs=red_xy,
                                              channel2_locs=green_xy,
                                              radii=radii)
    
    red_spearman = calc_spearman_cor_coeff(ch1_dist=rr_dist, ch2_dist=rg_dist)

    colocs_red = calc_coloc_values(spearman=red_spearman, ch1_locs=red_xy,
                                   ch2_locs=green_xy, radii=radii)
    
    save_locs_colocs(add_coloc_values(locs=red, coloc_values=colocs_red),
                     channel=2, out=out)
    
    all_locs = combine_channel_locs(add_coloc_values(locs=green, coloc_values=colocs),
                                    add_coloc_values(locs=red, coloc_values=colocs_red))

    save_locs_colocs(all_locs, channel=3, out=out)

def cluster_classification():

    print('Enter path to localisation file')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()

    data = load_locs(path=path, channels=2)

    clusters = hdbscan(locs=data, min_n=4)

    save_dbscan_results(data=clusters, n_channels=2, outpath=outpath)

    dbscan_filt = denoise_data(dbscan_data=clusters, min_n=4)

    save_dbscan_results(data=dbscan_filt, n_channels=2, outpath=outpath, filt=1)

def cluster_analysis_all():

    print('Enter path to DBSCAN data.')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()

    print('Cluster analysis has started.')
    
    dbscan_filt = load_dbscan_data(path=path)

    clust_analysed = analyse_clusters(dbscan_data=dbscan_filt)

    clust_filt = filter_clusters(cluster_data=clust_analysed)

    clust_filt_df = convert_to_dataframe(filt_cluster_data=clust_filt)

    save_cluster_analysis(filt_cluster_data=clust_filt_df, outpath=outpath)

    plot_cluster_statistics(filt_cluster_data=clust_filt_df, outpath=outpath)

    cluster_stats = calculate_statisitcs(filt_cluster_data=clust_filt_df)

    save_statistics(cluster_stats, out=outpath)

    print('Analysis complete')

def cluster_analysis_coloc():

    print('Enter path to DBSCAN data.')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()

def main():

    pass

if __name__ == '__main__':

    pass