import numpy as np
import matplotlib.pyplot as plt
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
import math

def user_input():

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

def load_locs(path: str):

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

    radii = list(np.arange(0, bounding_radius, increment))

    return radii[1:]

def ripley_k_function(xy_data, r, br):

    """
    2D Ripley's K=function. Converts result to numpy array
    """

    k = ripleyk.calculate_ripley(r, br, d1=xy_data[:, 0], d2=xy_data[:, 1])
    
    return np.array(k).reshape(len(k), 1)

def ripley_l_function(k_values):

    """
    2D Ripley's L-function, normalized such that the expected value is r
    """

    return np.sqrt(k_values / np.pi)

def ripley_h_function(l_values, radii):

    """
    2D Ripley's H-function, normalized such that the expected value is 0.
    The radii are converted to a numpy array.
    """

    return l_values - np.array(radii).reshape(len(radii), 1)

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
    ax.set_xlabel('Radius (nm)', labelpad=6, fontsize=40)
    ax.set_ylabel('H(r)', labelpad=2, fontsize=40)

    plt.savefig(out + '/' + str(title) + '.png')

def calculate_rmax(h_values, radii):

    return radii[h_values.argmax()]

def save_max_r(outpath, max_r):

    with open(outpath + '/max_r.txt', 'w') as f:

        f.write('The maximum value of r is: ' + str(max_r)
                + ' nm')

def hdbscan(locs, min_n):

    """"
    HDBSCAN clustering of localization data
    """

    # Instantiate and fit
    hdbscan = HDBSCAN(min_cluster_size=min_n).fit(locs[:, 2:4].reshape(-1, 2))

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

def save_dbscan_results(data, outpath):

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
    
    dbscan_results_df = pd.DataFrame(data=data, columns=cols)

    dbscan_results_df.to_csv(outpath + '/dbscan_output.csv')

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

    cluster_labels = np.unique(dbscan_data[:, -2])

    for label in cluster_labels:

        cluster_points = dbscan_data[(dbscan_data[:, 3] == label)]

        cluster_points_xy = cluster_points[:, 0:2]

        intensity = calculate_intensity(cluster_points_xy)

        center_of_mass = calculate_center_of_mass(cluster_points_xy)

        cluster_area = calculate_clust_area(cluster_points_xy)

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

    return filtered_clust_data[(filtered_clust_data[:, 3] > 20)]


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
                      filt_cluster_data.columns[i], out=outpath)

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
    """

    return locs[:, 2:4].reshape(-1, 2).astype(np.float32)

def calculate_transformation_matrix(channel1, channel2):

    """ Note: this function registers the first channel to the second channel
    I.e. it shifts the first channel to the second
    """

    M, inliers = cv.estimateAffinePartial2D(channel1, channel2)

    return M

def register_channel(channel, matrix):

    # Use the first channel

    corrected_channel = cv.transform(np.array([channel]), matrix)

    return corrected_channel.reshape(channel.shape[0], 2)

def compare_channels(channel1, channel2, out):

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 10

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    ax.scatter(channel1[:, 0], channel1[:, 1], s=20,
               facecolors='b')
    ax.scatter(channel2[:, 0], channel2[:, 1], s=20,
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

    plt.savefig(out + '/')

def save_corrected_channels(cor_locs, locs, out):

    cor_data = np.hstack((cor_locs, locs[:, 2:])).reshape(-1, 9)

    np.savetxt(out + '/cor_locs.csv', cor_data,
               fmt='%.6e', delimiter=',')

def add_channel(locs, channel: int):

    channel_col = np.repeat(channel, locs.shape[0]).reshape(locs.shape[0], 1)

    return np.hstack((locs, channel_col)).reshape(-1, 10)

def calc_counts_with_radius(locs, x0, y0, radii):

    """
    Calculate number of localisations from a list of increasing radii.
    """

    loc_counts_with_r = np.zeros((1, len(radii)))

    for i in range(0, len(radii)):

        filtered_locs = locs[((locs[:, 0] - x0)**2 + 
                                (locs[:, 1] - y0)**2 < radii[i]**2)]
        
        loc_counts_with_r[0, i] = filtered_locs.shape[0] + 1
    
    return loc_counts_with_r

def calc_loc_distribution(counts, radii):

    """
    Calculates distribution of number of localisations with increasing radii
    from a localisation
    """

    max_r = max(radii)

    cbc = counts / np.max(counts) * (max_r ** 2 / np.array(radii) ** 2)

    return cbc

def calc_all_distributions(channel1_locs, channel2_locs, radii):

    """
    Combines the previous two functions to calculate distributions along
    an increasing radius for all distributions of a particular channel.
    """

    dist_ch1 = []

    dist_ch2 = []

    # Loop through all localisations

    for i in range(0, channel1_locs.shape[0]):

        x0, y0 = channel1_locs[i, 0], channel1_locs[i, 1]

        # Channel 1
        ch1_counts = calc_counts_with_radius(
            locs=channel1_locs, x0=x0, y0=y0, radii=radii
        )

        dist_ch1.append(
            calc_loc_distribution(counts=ch1_counts, radii=radii)
        )

        # Channel 2
        ch2_counts = calc_counts_with_radius(
            locs=channel2_locs, x0=x0, y0=y0, radii=radii
        )

        dist_ch2.append(
            calc_loc_distribution(counts=ch2_counts, radii=radii)
        )
    
    return np.vstack(dist_ch1).reshape(-1, len(radii)), np.vstack(dist_ch2).reshape(-1, len(radii))

def calc_pearson_cor_coeff(ch1_dist, ch2_dist):

    """
    Calculate Pearson correlation coefficients on a row-by-row basis
    for the distributions of channel 1 to itself and to channel 2. I.e, 
    between D_AA(r) and D_AB(r)
    """

    pearson_cor_coeffs = np.zeros((ch1_dist.shape[0], 1))

    pearson_cor_coeffs = stats.pearsonr(ch1_dist, ch2_dist, axis=1).statistic

    return pearson_cor_coeffs.reshape(ch1_dist.shape[0], 1)

def add_pearson_coeffs(locs, cor_coeffs):

    """
    Add correlation coefficients to localisation data.
    """

    return np.hstack((locs, cor_coeffs)).reshape(-1, 11)

def save_locs_pearsons(processed_data, out):

    """
    Save localisations with correlation coefficients as .csv file.
    """

    np.savetxt(out + '/locs_with_pearson.csv',
               processed_data, fmt='%.6e', delimiter=',')

def test_ripley_clustering():

    print('Enter path to localisation file')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()

    print('Enter bounding radius.')
    bound_r = float(user_input())

    print('Enter increment.')
    increment_r = float(user_input())
    
    data = load_locs(path)

    xy = extract_xy(data)
    
    radii = generate_radii(bounding_radius=bound_r,
                           increment=increment_r)
    
    k_values = ripley_k_function(xy, r=radii, br=bound_r)

    l_values = ripley_l_function(k_values=k_values)

    h_values = ripley_h_function(l_values=l_values, radii=radii)

    plot_ripley_h(h_values=h_values, radii=radii, out=outpath, title='h_function')

    rmax = calculate_rmax(h_values=h_values, radii=radii)

    save_max_r(outpath=outpath, max_r=rmax)

def test_hdbscan():

    print('Enter path to localisation file')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()

    data = load_locs(path=path)

    clusters = hdbscan(locs=data, min_n=4)

    save_dbscan_results(data=clusters, outpath=outpath)

    dbscan_filt = denoise_data(dbscan_data=clusters, min_n=4)

    save_dbscan_results(data=dbscan_filt, outpath=outpath)

    clust_analysed = analyse_clusters(dbscan_data=dbscan_filt)

    clust_filt = filter_clusters(cluster_data=clust_analysed)

    clust_filt_df = convert_to_dataframe(filt_cluster_data=clust_filt)

    save_cluster_analysis(filt_cluster_data=clust_filt_df, outpath=outpath)

    plot_cluster_statistics(filt_cluster_data=clust_filt_df, outpath=outpath)

def main():

    pass

if __name__ == '__main__':

    test_hdbscan()