import numpy as np
import pandas as pd
import os

def load_locs(path: str, channels: int=1) -> 'np.ndarray[np.float32]':

    """
    Extract localisation data from .csv file, preferably from ThunderSTORM
    
    In: path (str)---file path to localization data

    Out: locs---localization data from ThunderSTORM (numpy array)
    
    """

    locs = pd.read_csv(path, sep=',', header=None,
                               engine='pyarrow', skiprows=1)
    
    locs = np.array(locs).astype(np.float32)
    
    if channels == 1:
    
        return locs.reshape(-1, 9)
    
    else:

        return locs.reshape(-1, 11)

def extract_xy(locs: 'np.ndarray[np.float32]') -> 'np.ndarray[np.float32]': 

    """
    Extract xy localisations.

    In: localization table (numpy array)

    Out: xy localizations (numpy array)
    """

    return locs[:, 2:4].reshape(-1, 2)

def save_reg_error(outpath: str, reg_error: float) -> None:

    """
    Save the registration error from channel registration
    """

    with open(os.path.join(outpath, "registration_error.txt"), "w") as f:

        f.write("The channel registration error is: " + str(reg_error) + " nm")

def save_max_r(outpath: str, max_r: float) -> None:

    """
    Save the value of the radius at which Ripley's H-function is at a maximum
    """

    with open(os.path.join(outpath, "max_r.txt"), 'w') as f:

        f.write('The maximum value of r is: ' + str(max_r)
                + ' nm')

def save_corrected_channels(cor_locs: 'np.ndarray[np.float64]', locs:'np.ndarray[np.float64]', out: str):

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

    locs_df.to_csv(os.path.join(out, "corrected_locs_ch1.csv"), index=False)
    
    return cor_data

def save_locs_colocs(processed_data: 'np.ndarray[np.float32]', channel: int, out: str):

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

    locs_proc.to_csv(os.path.join(out, "processed_locs_" +
                                  str(channel) + '.csv'), index=False)

##-------DBSCAN FILE IO--------##

def save_dbscan_results(data: 'np.ndarray[np.float64]', n_channels: int, outpath: str, filt: int=0):

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
    
        dbscan_results_df.to_csv(os.path.join(outpath, 'dbscan_output.csv'), index=False)

    else:

        dbscan_results_df.to_csv(os.path.join(outpath, 'filt_dbscan_output.csv'), index=False)

def save_cluster_analysis(filt_cluster_data: 'pd.DataFrame', outpath: str):

    """
    Save results of cluster analysis as a .csv file.

    In: filt_cluster_data---cluster statistics following filtering (pd DataFrame)
    outpath---folder where data will be saved (str)
    coloc---specifies whether all data (coloc=0), non-colocalised data (coloc=1),
    or colocalised data (coloc=2) is to be saved (int)

    Out: None but a .csv file with cluster stats is saved.
    """

    title = 'cluster_statistics.csv'

    filt_cluster_data.to_csv(os.path.join(outpath, title), sep=',',
                                 index=False)

def save_statistics(cluster_statistics: dict, out: str, coloc: int=0):

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

        with open(os.path.join(out, "cluster_stats_summary.txt"), 'w') as f:
            
            for stat in cluster_statistics:

                print(stat + ' ' + str(cluster_statistics[stat]) + '\n',
                    file=f)
    
    elif coloc == 1:

        with open(os.path.join(out, "cluster_stats_summary_no_coloc.txt"), 'w') as f:
            
            for stat in cluster_statistics:

                print(stat + ' ' + str(cluster_statistics[stat]) + '\n',
                    file=f)
    
    else:

        with open(os.path.join(out, "cluster_stats_summary_coloc.txt"), 'w') as f:
            
            for stat in cluster_statistics:

                print(stat + ' ' + str(cluster_statistics[stat]) + '\n',
                    file=f)

##------CLUSTER ANALYSIS FILE IO-------##

def collate_clust_files(folder: str) -> list[str]:

    """
    Collates .csv files with cluster statistics and returns the file paths
    as a list.

    In: folder. The folder where cluster statistics files are stored.
    
    Out: clust_files - file paths of cluster data, sorted alphabetically.
    """

    clust_files = [
        os.path.join(folder, file) for file in os.listdir(folder)
        if file.endswith(".csv")
    ]

    return sorted(clust_files)

def load_clust_stats(file: str) -> 'pd.DataFrame':

    """
    Loads a .csv file from its file path, using only columns 2,3,4,5, and 6.
    Note that these correspond to cluster properties.

    In: file - .csv file containing the cluster data.
    
    Out: clust_stats - pandas dataframe containing cluster properties.
    """

    cols = ['Area (nm^2)',
        'Radius (nm)',
        'Circularity',
        'Number of Channel 1',
        'Density (n . um^-2)']

    clust_stats = pd.read_csv(file, sep=",", usecols=cols, engine='pyarrow')

    return clust_stats

def combine_all_stats(file_paths: list[str], conditions: list[str]) -> "pd.DataFrame":

    """
    Loads all .csv files of cluster statistics, inserts a new column
    corresponding to the user-supplied specifications for experimental
    conditions, and returns a single dataframe that concatenates all
    data.

    In: file_paths, a list of file paths for the cluster statistics.
    conditions: user-supplied list of experimental conditions.
    
    Out: all_data - pandas dataframe containing all cluster properties
    for all experimental conditions.
    """

    all_data = []

    for file_path, condition in zip(file_paths, conditions):

        clust_data = load_clust_stats(file_path)

        cond_list = [condition] * clust_data.shape[0]

        clust_data.insert(clust_data.shape[1], "Condition", cond_list)

        all_data.append(clust_data)
    
    return pd.concat(all_data)

def save_expl_var(variance_ratio: "np.ndarray", out: str) -> None:

    with open(os.path.join(out, "explained_variance.txt"), "w") as f:

        for i, ratio in enumerate(variance_ratio):

            f.write("The explained variance is " + str(ratio) + 
                    " for principal component " + str(i) + "\n")