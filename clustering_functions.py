import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import HDBSCAN
import ripleyk
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
import math


def load_locs(path):

    """
    Extract localisation data from .csv file, preferably from ThunderSTORM
    """

    locs = np.genfromtxt(path, delimiter=',', dtype=float,
                         skip_header=1)
    
    return locs.reshape(-1, 9)

def extract_xy(locs):

    """
    Extract xy localisations
    """

    return locs[:, 2:4]

def generate_radii(bounding_radius, increment):

    """
    Generate a list of radii for cluster detection via Ripley functions.
    Bounding radius is the maximum radius while the increment defines the range 
    of radii for the functions.
    """

    return list(range(0, bounding_radius, increment))

def ripley_k_function(xy_data, r, br):

    """
    2D Ripley's K=function
    """

    k = ripleyk.calculate_ripley(r, br, d1=xy_data[:, 0], d2=xy_data[:, 1],
                                 boundary_correct=True, CSR_Normalise=True)
    
    return k

def ripley_l_function(k_values):

    """
    2D Ripley's L-function, normalized such that the expected value is r
    """

    return math.sqrt(k_values / np.pi)

def ripley_h_function(l_values, radii):

    """
    2D Ripley's H-function, normalized such that the expected value is 0.
    """

    return l_values - radii

def plot_ripley_h(h_values, radii, out, title):

    """
    Plots Ripley's H-function against radii.
    """

    plt.ioff()

    # Set font type and size for axis values
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(12, 12), dpi=500)

    # Plot Ripley's H-function against radii
    ax.plot(radii, h_values, 'b', linewidth=5.0)

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
    ax.set_xlabel('Radius (nm)', labelpad=6, fontsize=40)
    ax.set_ylabel('H(r)', labelpad=2, fontsize=40)

    plt.savefig(out + '/' + str(title) + '.png')


def hdbscan(locs, min_n):

    """"
    HDBSCAN clustering of localization data
    """

    # Instantiate and fit
    hdbscan = HDBSCAN(min_cluster_size=min_n).fit(locs[:, 2:4])

    # DBSCAN labels
    labels = hdbscan.labels_

    # Reassgine label for noise
    labels[(labels == 0)] = -1

    cluster_probabilities = hdbscan.probabilities_

    # Combine localisation data, labels, and probabilities
    return np.concatenate((locs, labels[:, np.newaxis], cluster_probabilities[:, np.newaxis]), axis=1)

def denoise_data(dbscan_data, min_n):

    """
    Removes clusters below a minimum localizations threshold and noise.
    """

    # Remove noise
    noiseless_data = dbscan_data[(dbscan_data[:, -2] > 0)]

    # Count how many localisations are assigned to each cluster
    label_count = dict(Counter(noiseless_data[:, -2]))

    for label, count in label_count.items():

        # If no. of localizations is below the threshold, discard from data
        if count < min_n:
                
            noiseless_data = noiseless_data[(noiseless_data[:, -2] != float(label))]
        
        else:
            
            continue
    
    return noiseless_data

def save_dbscan_results(filtered_data, outpath):

    """
    Convert dbscan results to dataframe and save as .csv
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
            'label',
            'probability']
    
    dbscan_results_df = pd.DataFrame(data=filtered_data, columns=cols)

    dbscan_results_df.to_csv(outpath + '/dbscan_output.txt')

def calculate_intensity(points):

    """
    Calculates cluster intensity, i.e. how many localisations per cluster.
    """

    return np.size(points, axis=0)

def calculate_center_of_mass(points):

    """
    Calculates cluster centroid.
    """

    return np.mean(points, axis=0)

def calculate_clust_area(points):

    """
    Calculates cluster area with the Convexhull method.
    """

    return ConvexHull(points).volume

def calculate_radius(points, center):

    """
    Radius calculation. First calculates the pairwise distance of all
    cluster points from the centroid then selects the maximum.
    """
    
    return np.max(pairwise_distances(points, center))

def analyse_clusters(dbscan_data):

    """
    This function loops through each cluster label, extracts the xy localisations,
    then calculates cluster intensity, area, and radius.
    """

    analysis_results = []

    cluster_labels = np.unique(dbscan_data[:, 3])

    for label in cluster_labels:

        cluster_points = dbscan_data[(dbscan_data[:, 3] == label)]

        cluster_points_xy = cluster_points[:, 0:2]

        intensity = calculate_intensity(cluster_points)

        center_of_mass = calculate_center_of_mass(cluster_points)

        cluster_area = calculate_clust_area(cluster_points)

        center = center_of_mass[:, np.newaxis]
        center = center.T

        cluster_radius = calculate_radius(cluster_points_xy, center=center)

        analysis_results.append([center_of_mass[0], center_of_mass[1], cluster_area,
                                 cluster_radius, intensity, label])
        
    return np.array(analysis_results).reshape(-1, 6)

def filter_clusters(cluster_data):

    """
    Remove clusters with very large radii and ensure no values are nan.
    """
    cluster_data = cluster_data[~np.isnan(cluster_data).any(axis=1), :]

    filtered_clust_data = cluster_data[(cluster_data[:, 3] < 400)]

    return filtered_clust_data[(filtered_clust_data[:, 3] > 0)]


def convert_to_dataframe(filt_cluster_data):

    """
    Converts cluster analysis results to a dataframe.
    """

    cols = [
        'x[nm]',
        'y[nm]',
        'Area [nm^2]',
        'Radius [nm]',
        'Intensity',
        'label'
    ]

    cluster_data_df = pd.DataFrame(data=filt_cluster_data, columns=cols)

    return cluster_data_df
    
def save_cluster_analysis(filt_cluster_data, outpath):

    filt_cluster_data.to_csv(outpath + '/dbscan_analysis.csv', sep=',')

def plot_histogram(data, title, out):

    """
    Plots a histogram for a column of data.
    """

    plt.ioff()

    weights = np.ones_like(data) / float(len(data))

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

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

    ax.set_xlabel(title, labelpad=6, fontsize=40)
    ax.set_ylabel('Frequency', labelpad=2, fontsize=40)

    plt.savefig(out + '/' + title + '.png')

def plot_cluster_statistics(filt_cluster_data, outpath):

    """
    Plots and saves histograms for cluster intensity, area, and radius.
    """

    for i in range(2, 5):

        plot_histogram(filt_cluster_data[filt_cluster_data.columns[i]],
                      filt_cluster_data.columns[1], out=outpath)

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

    ax.scatter()
    
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


def main():

    path = 'C:/users/mxq76232/Downloads/pictest/rip_test.csv'

    data = load_locs(path)

    xy = extract_xy(data)

    radii = generate_radii(bounding_radius=1000, increment=100)

    value = ripley_h_function(xy, r=radii, br=1000)

    print(value)

if __name__ == '__main__':

    main()