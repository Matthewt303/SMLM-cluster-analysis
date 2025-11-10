import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import os

def plot_reg_error(nneighbors: "np.ndarray", out: str):

    plt.ioff()

    weights = np.ones_like(nneighbors) / float(len(nneighbors))

    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 24

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    plt.hist(nneighbors, bins=15, weights=weights, edgecolor='black',
             linewidth=2.0, color='darkred')

    ax.set_xlim(right=np.max(nneighbors) + 0.05 * np.max(nneighbors))
    
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
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

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

    ax.set_xlabel("Nearest neighbour distance (nm)", labelpad=6, fontsize=28)
    ax.set_ylabel('Normalized frequency', labelpad=1, fontsize=28)

    plt.savefig(os.path.join(out, '_reg_error_plot.png'))
    plt.savefig(os.path.join(out, '_reg_error_plot.svg'))

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

    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 28

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    ax.plot(radii, h_values / 100, 'darkblue', linewidth=5.0, label="H(r)")
    ax.axhline(y=0, color='darkred', linewidth=4.5, label="H(r) = 0")

    leg = plt.legend(bbox_to_anchor=(0.5, 1.125), loc="upper center", ncol=2)

    for line in leg.get_lines():
        line.set_linewidth(3.5)

    for text in leg.get_texts():
        text.set_fontsize(28)

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
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

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
    ax.set_xlabel('Radius (nm)', labelpad=6, fontsize=36)
    ax.set_ylabel('H(r) x 100', labelpad=2, fontsize=36)

    plt.savefig(os.path.join(out, "ripley_h_func.svg"))
    plt.savefig(os.path.join(out, "ripley_h_func.png"))

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

    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 24

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    plt.hist(data, bins=20, weights=weights, edgecolor='black',
             linewidth=2.0, color='darkred')

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
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

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

    ax.set_xlabel(title, labelpad=6, fontsize=28)
    ax.set_ylabel('Normalized frequency', labelpad=1, fontsize=28)

    if coloc == 0:
    
        plt.savefig(os.path.join(out, title + '.png'))
        plt.savefig(os.path.join(out, title + '.svg'))
    
    elif coloc == 1:

        plt.savefig(os.path.join(out, title + '_no_coloc.png'))
        plt.savefig(os.path.join(out, title + '_no_coloc.svg'))
    
    else:

        plt.savefig(os.path.join(out, title + '_coloc.png'))
        plt.savefig(os.path.join(out, title + '_coloc.svg'))

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

##-----PCA plots-----##

def plot_components_2d(final_df: 'pd.DataFrame', out: str) -> None:

    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 24
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)
    
    conditions = set(final_df[final_df.columns[-1]])
    
    colors = iter(plt.cm.magma(np.linspace(0, 1, final_df.shape[1])))
    
    for condition in conditions:
        
        indices = final_df[final_df.columns[-1]] == condition
        
        ax.scatter(final_df.loc[indices, 'PC1 Reduced Data'], 
                   final_df.loc[indices, 'PC2 Reduced Data'],
                   c=next(colors), s=40, alpha=0.5, label=condition)
    
    leg = plt.legend(bbox_to_anchor=(0.5, 1.175), loc="upper center", ncol=2)

    for line in leg.get_lines():
        line.set_linewidth(3.5)

    for text in leg.get_texts():
        text.set_fontsize(28)
    
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
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

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
    ax.set_xlabel('Principal Component 1', labelpad=6, fontsize=32)
    ax.set_ylabel('Principal Component 2', labelpad=2, fontsize=32)

    plt.savefig(os.path.join(out, "pca_plot.svg"))
    plt.savefig(os.path.join(out, "pca_plot.png"))
        

def plot_components_3d(final_df: 'pd.DataFrame'):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    targets = set(final_df[final_df.columns[-1]])
    
    colors = iter(plt.cm.viridis(np.linspace(0, 1, final_df.shape[1])))
    
    for target in targets:
        
        indices = final_df[final_df.columns[-1]] == target
    
        ax.scatter(final_df.loc[indices, 'Principal component 1'],
                   final_df.loc[indices, 'Principal component 2'],
                   final_df.loc[indices, 'Principal component 3'],
                   c=next(colors), s=60, label=target)
    
    plt.show()
    
def plot_var_ratio(var_ratio: 'np.ndarray[np.float64]', out: str):

    """

    """

    plt.ioff()

    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 24

    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)

    ax.plot(["PC1", "PC2", "PC3", "PC4", "PC5"], var_ratio, 'darkblue', linewidth=5.0, label="H(r)")

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
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

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
    ax.set_xlabel('Principal components', labelpad=2, fontsize=32)
    ax.set_ylabel('Ratio of variance', labelpad=2, fontsize=32)

    plt.savefig(os.path.join(out, "var_ratio.svg"))
    plt.savefig(os.path.join(out, "var_ratio.png"))