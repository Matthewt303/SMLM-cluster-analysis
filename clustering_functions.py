import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib as mpl


## Cluster visualisation

def make_circles(x: 'np.ndarray[np.float64]', y: 'np.ndarray[np.float64]', r: 'np.ndarray[np.float64]') -> list:

    circles = [plt.Circle((xi, yi), radius=c, linewidth=0)
               for xi, yi, c in zip(x, y, r)]
    
    return circles

def extract_xyr(cluster_data: 'np.ndarray[np.float64]') -> 'np.ndarray[np.float64]':

    x = cluster_data[cluster_data.columns[0]]

    y = cluster_data[cluster_data.columns[1]]

    r = cluster_data[cluster_data.columns[3]]

    return x, y, r

def scale(data: 'np.ndarray[np.float64]', scaling_factor: float) -> 'np.ndarray[np.float64]':

    return data / scaling_factor

def plot_clusters(cluster_data: 'np.ndarray[np.float64]', loc_data: 'np.ndarray[np.float64]', out: str, title: str):

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
