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
    
    return locs

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

def ripley_h_function(xy_data, r, br):

    k = ripleyk.calculate_ripley(r, br, d1=xy_data[:, 0], d2=xy_data[:, 1],
                                 boundary_correct=True, CSR_Normalise=True)
    
    return k

def ripley_l_function(k_values):

    return math.sqrt(k_values / np.pi)

def ripley_h_function(l_values, radii):

    return l_values - radii

def plot_ripley_h(h_values, radii, out, title):

    plt.ioff()

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(12, 12), dpi=500)

    ax.plot(radii, h_values, 'b', linewidth=5.0)

    ax.set_xlim(left=0)

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

    ax.set_xlabel('Radius (nm)', labelpad=6, fontsize=40)
    ax.set_ylabel('H(r)', labelpad=2, fontsize=40)

    plt.savefig(out + '/' + str(title) + '.png')


def hdbscan(locs, min_n):

    hdbscan = HDBSCAN(min_cluster_size=min_n).fit(locs[:, 0:2])

    labels = hdbscan.labels_

    labels[(labels == 0)] = -1

    cluster_probabilities = hdbscan.probabilities_

    return np.concatenate((locs[:, 0:3], labels[:, np.newaxis], cluster_probabilities[:, np.newaxis]), axis=1)

def denoise_data(dbscan_data, min_n):

    noiseless_data = dbscan_data[(dbscan_data[:, 3] > 0)]

    label_count = dict(Counter(noiseless_data[:, 3]))

    for label, count in label_count.items():

        if count < min_n:
                
            noiseless_data = noiseless_data[(noiseless_data[:, 3] != float(label))]
        
        else:
            
            continue
    
    return noiseless_data

def save_dbscan_results(filtered_data, outpath, min_n):

    cluster_labels = np.unique((filtered_data[:, 3]))

    number_of_clusters = np.size(cluster_labels)

    np.savetxt(outpath + "/dbscan_output.txt", cluster_labels, fmt='%.5e', 
               header="DBSCAN \n x[nm] y[nm] t[frame] cluster \n "
                " Number of clusters = " + str(number_of_clusters) + "\n"
                 +  "n = " + str(min_n))

def calculate_intensity(points):

    return np.size(points, axis=0)

def calculate_center_of_mass(points):

    return np.mean(points, axis=0)

def calculate_clust_area(points):

    return ConvexHull(points).volume

def calculate_radius(points, center):
    
    return np.max(pairwise_distances(points, center))

def analyse_clusters(dbscan_data):

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

def filter_clusters():

    pass


def convert_to_dataframe(cluster_data):

    cols = [
        'x[nm]',
        'y[nm]',
        'area[nm^2]',
        'radius [nm]',
        'intensity',
        'label'
    ]

    cluster_data_df = pd.DataFrame(data=cluster_data, columns=cols)

    return cluster_data_df
    
def save_cluster_analysis(cluster_data, outpath):

    cluster_data.to_csv(outpath + "/dbscan_analysis.txt", sep='\t')

def main():

    path = 'C:/users/mxq76232/Downloads/pictest/rip_test.csv'

    data = load_locs(path)

    xy = extract_xy(data)

    radii = generate_radii(bounding_radius=1000, increment=100)

    value = ripley_h_function(xy, r=radii, br=1000)

    print(value)

if __name__ == '__main__':

    main()