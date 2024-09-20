import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import OPTICS, DBSCAN
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances
import math

import ripleyk

def load_locs(path):

    locs = np.genfromtxt(path, delimiter=',', dtype=float,
                         skip_header=1)
    
    return locs

def extract_xy(locs):

    return locs[:, 2:4]

def generate_radii(bounding_radius, increment):

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

def optics_clustering(xy_data, min_samples):

    clusters = OPTICS(min_samples=min_samples, max_eps=100,
                          n_jobs=1).fit(xy_data)
    
    return clusters.core_distances_

def calculate_epsilon(optics_distances, threshold):

    filtered_distances = optics_distances[(optics_distances < 200)]

    distances_count = Counter(filtered_distances)

    sorted_counts = dict(sorted(distances_count.items()))

    norm_distance_counts = np.array(list(sorted_counts.values())) / sum(sorted_counts.values())

    cumulative_distances = np.cumsum(norm_distance_counts)

    cumulative_distances = cumulative_distances[:, np.newaxis]

    distances = np.array(list(sorted_counts.keys()))

    distances = distances[:, np.newaxis]

    distances_vs_counts = np.concatenate((distances, cumulative_distances), axis=1)

    epsilon = np.min(distances_vs_counts[(distances_vs_counts[:, 1]) >= 0.9], axis=0)

    return epsilon[0]

def save_eps(outpath, epsilon):

        with open(outpath + '/optics_epsilon.txt', 'w') as f:
            f.write('The value of epsilon is: ' + str(epsilon) + ' nm')

def dbscan(locs, epsilon, min_n):

    dbscan = DBSCAN(eps=epsilon, min_samples=min_n).fit(locs[:, 0:2])

    labels = dbscan.labels_

    labels[(labels == 0)] = -1

    return np.concatenate((locs[:, 0:3], labels[:, np.newaxis],), axis=1)

def denoise_data(dbscan_data, min_n):

    noiseless_data = dbscan_data[(dbscan_data[:, 3] > 0)]

    label_count = dict(Counter(noiseless_data[:, 3]))

    for label, count in label_count.items():

        if count < min_n:
                
            noiseless_data = noiseless_data[(noiseless_data[:, 3] != float(label))]
        
        else:
            
            continue

def save_dbscan_results(filtered_data, outpath, epsilon, min_n):

    cluster_labels = np.unique((filtered_data[:, 3]))

    number_of_clusters = np.size(cluster_labels)

    np.savetxt(outpath + "/dbscan_output.txt", cluster_labels, fmt='%.5e', 
               header="DBSCAN \n x[nm] y[nm] t[frame] cluster \n "
                " Number of clusters = " + str(number_of_clusters) +
                " \n Epsilon = " + str(epsilon) + " nm" + " n = " + str(min_n))

def main():

    path = 'C:/users/mxq76232/Downloads/pictest/rip_test.csv'

    data = load_locs(path)

    xy = extract_xy(data)

    radii = generate_radii(bounding_radius=1000, increment=100)

    value = ripley_h_function(xy, r=radii, br=1000)

    print(value)

if __name__ == '__main__':

    main()