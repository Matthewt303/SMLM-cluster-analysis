import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import numpy as np

def plot_ripley_h(h_values: 'np.ndarray[np.float64]', radii: list[float], out: str):

    """
    Plots Ripley's H-function against radii.

    In: h_values---Ripley h-values (numpy array)
    radii---list of radii over which Ripley's K-function was calculated (np array)
    out---folder path where plot will be saved (str)
    title---the file name of the plot (str)

    Out: saves .png of H-values against radii.
    """

    plt.ioff()

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 10

    fig, ax = plt.subplots(figsize=(8, 8), dpi=500)

    ax.plot(radii, h_values, 'b', linewidth=5.0)
    ax.axhline(y=0, color='r')

    ax.set_xlim(left=0)

    ratio = 1.0

    # Make sure figure is square
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

    # Axis labels
    ax.set_xlabel('Radius (nm)', labelpad=6, fontsize=20)
    ax.set_ylabel('H(r)', labelpad=2, fontsize=20)

    plt.savefig(out + '/ripley_hfunction.png')

def plot_bar_percent_coloc(percent_ch1: float, percent_ch2: float,
                           name_ch1: str, name_ch2: str, out: str):

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
    mpl.rcParams['font.size'] = 11

    fig, ax = plt.subplots(figsize=(8, 8), dpi=500)

    ax.bar([name_ch1, name_ch2], [percent_ch1, percent_ch2],
           color=['midnightblue', 'darkred'])

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

    ax.set_ylabel('Colocalised molecules (%)', labelpad=12, fontsize=20)

    plt.savefig(out + '/coloc_percent.png')

def plot_histogram(data: pd.DataFrame, title: str, out: str, coloc: int=0):

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
    mpl.rcParams['font.size'] = 11

    fig, ax = plt.subplots(figsize=(8, 8), dpi=500)

    plt.hist(data, bins=20, weights=weights, edgecolor='black',
             linewidth=1.1, color='darkred')

    ax.set_xlim(right=np.max(data) + 0.05 * np.max(data))
    
    ratio = 1.0

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)) * ratio)

    ax.tick_params(axis='y', which='major', length=6, direction='in')
    ax.tick_params(axis='y', which='minor', length=3, direction='in')
    ax.tick_params(axis='x', which='major', length=6, direction='in')
    ax.tick_params(axis='x', which='minor', length=3, direction='in')

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

    ax.set_xlabel(title, labelpad=6, fontsize=20)
    ax.set_ylabel('Frequency', labelpad=6, fontsize=20)

    if coloc == 0:
    
        plt.savefig(out + '/' + title + '.png')
    
    elif coloc == 1:

        plt.savefig(out + '/' + title + '_no_coloc.png')
    
    else:

        plt.savefig(out + '/' + title + '_coloc.png')

def plot_cluster_statistics(filt_cluster_data: pd.DataFrame, outpath: str, coloc: int=0):

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

def plot_boxplot(data: list, statistic: str, out: str):

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
    mpl.rcParams['font.size'] = 11

    medianprops = dict(linestyle='-', linewidth=1.5, color='midnightblue')
    boxprops = dict(linestyle='-', linewidth=1.5, color='black')
    whiskerprops = dict(linestyle='-', linewidth=1.5, color='black')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=500)

    ax.boxplot(data, medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
               showfliers=False)

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

    ax.set_xticks([i + 1 for i in range(len(data))],
                  labels=['Non-colocalised', 'Colocalised'])

    ax.set_ylabel(statistic, labelpad=8, fontsize=20)

    plt.savefig(out + '/' + statistic + '_boxplot.png')
