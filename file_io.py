import numpy as np
import pandas as pd

def user_input() -> str:

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

def load_locs(path: str, channels: int=1) -> 'np.ndarray[np.float64]':

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

def extract_xy(locs: 'np.ndarray[np.float64]') -> 'np.ndarray[np.float64]': 

    """
    Extract xy localisations.

    In: localization table (numpy array)

    Out: xy localizations (numpy array)
    """

    return locs[:, 2:4].reshape(-1, 2)

def save_max_r(outpath: str, max_r: float) -> None:

    """
    Save the value of the radius at which Ripley's H-function is at a maximum
    """

    with open(outpath + '/max_r.txt', 'w') as f:

        f.write('The maximum value of r is: ' + str(max_r)
                + ' nm')

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
    
        dbscan_results_df.to_csv(outpath + '/dbscan_output.csv', index=False)

    else:

        dbscan_results_df.to_csv(outpath + '/filt_dbscan_output.csv', index=False)

def save_cluster_analysis(filt_cluster_data: pd.DataFrame, outpath: str, coloc: int=0):

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

        filt_cluster_data.to_csv(outpath + '/cluster_analysis_no_coloc.csv', sep=',',
                                 index=False)
    
    else:

        filt_cluster_data.to_csv(outpath + '/cluster_analysis_coloc.csv', sep=',',
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

        with open(out + '/cluster_stats_summary.txt', 'w') as f:
            
            for stat in cluster_statistics:

                print(stat + ' ' + str(cluster_statistics[stat]) + '\n',
                    file=f)
    
    elif coloc == 1:

        with open(out + '/cluster_stats_summary_no_coloc.txt', 'w') as f:
            
            for stat in cluster_statistics:

                print(stat + ' ' + str(cluster_statistics[stat]) + '\n',
                    file=f)
    
    else:

        with open(out + '/cluster_stats_summary_coloc.txt', 'w') as f:
            
            for stat in cluster_statistics:

                print(stat + ' ' + str(cluster_statistics[stat]) + '\n',
                    file=f)