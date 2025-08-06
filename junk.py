## Main functions

def two_color_reg_accuracy():

    """
    Checks the accuracy of channel registration using two localisation
    tables from beads.
    """

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

    nearest_neighbors = calculate_nneighbor_dist(ch1_locs=green_xy_reg, ch2_locs=red_bead_xy, radii=[1, 1])

    green_xy_filt, red_xy_filt = filter_bead_locs(ch1_locs=green_bead_xy, ch2_locs=red_bead_xy, nneighbors=nearest_neighbors)

    matrix_2 = calculate_transformation_matrix(channel1=green_xy_filt, channel2=red_xy_filt)

    green_xy_reg_2 = register_channel(channel=green_xy_filt, matrix=matrix_2)

    compare_channels(green_xy_reg_2, red_bead_xy)

    nearest_neighbors_2 = calculate_nneighbor_dist(ch1_locs=green_xy_reg_2, ch2_locs=red_bead_xy, radii=[1, 1])

    plot_histogram(data=nearest_neighbors_2, title='reg_accuracy', out=out)

    print(np.median(nearest_neighbors))

    print(np.median(nearest_neighbors_2))

    print(matrix)

    print(matrix_2)

def cluster_classification():

    """
    Carries out HDBSCAN, filters clusters, and appends the cluster
    label to the localisation table.
    """

    print('Enter path to localisation file')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()

    data = load_locs(path=path, channels=2)

    clusters = hdbscan(data, min_n=4)

    save_dbscan_results(clusters, n_channels=2, outpath=outpath)

    dbscan_filt = denoise_data(clusters, min_n=4)

    save_dbscan_results(data=dbscan_filt, n_channels=2, outpath=outpath, filt=1)

def cluster_analysis_all():

    """
    Calculates cluster statistics for all localisations. Saves histograms
    of each statistic, as well as medians and standard deviations.
    """

    print('Enter path to DBSCAN data.')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()

    print('Cluster analysis has started.')
    
    dbscan_filt = load_dbscan_data(path=path)

    #dbscan_filt = denoise_data(dbscan_data=dbscan_filt, min_n=4)

    clust_analysed = analyse_clusters(dbscan_filt)

    clust_filt = filter_clusters(clust_analysed)

    clust_filt_df = convert_to_dataframe(filt_cluster_data=clust_filt)

    save_cluster_analysis(filt_cluster_data=clust_filt_df, outpath=outpath)

    plot_cluster_statistics(filt_cluster_data=clust_filt_df, outpath=outpath)

    cluster_stats = calculate_statistics(filt_cluster_data=clust_filt_df)

    save_statistics(cluster_stats, out=outpath)

    print('Analysis complete')

def cluster_analysis_coloc():

    """
    Takes data from HDBSCAN, separates it by degree of colocalisation
    and carries out cluster analysis separately on each dataset. The function
    also plots boxplots to compare statistics between colocalised and non-colocalised.
    The result from a Mann-Whitney U test is also saved.
    """

    print('Enter path to DBSCAN data.')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()

    print('Cluster analysis has started.')

    dbscan_filt = load_dbscan_data(path=path)

    ch1_percent, ch2_percent = calc_percent_coloc(dbscan_data=dbscan_filt)

    plot_bar_percent_coloc(percent_ch1=ch1_percent, percent_ch2=ch2_percent,
                           name_ch1='ACE-2', name_ch2='Spike protein', out=outpath)

    no_coloc, coloc = separate_coloc_data(dbscan_data=dbscan_filt)

    no_coloc_filt, coloc_filt = denoise_data(no_coloc, min_n=4), denoise_data(coloc, min_n=4)

    no_coloc_analysed, coloc_analysed = analyse_clusters(dbscan_data=no_coloc_filt), analyse_clusters(dbscan_data=coloc_filt)

    no_coloc_analysed_filt, coloc_analysed_filt = filter_clusters(cluster_data=no_coloc_analysed), filter_clusters(coloc_analysed)

    no_coloc_df, coloc_df = convert_to_dataframe(filt_cluster_data=no_coloc_analysed_filt), convert_to_dataframe(coloc_analysed_filt)

    save_cluster_analysis(filt_cluster_data=no_coloc_df, outpath=outpath, coloc=1)

    save_cluster_analysis(filt_cluster_data=coloc_df, outpath=outpath, coloc=2)

    plot_cluster_statistics(filt_cluster_data=no_coloc_df, outpath=outpath, coloc=1)

    plot_cluster_statistics(filt_cluster_data=coloc_df, outpath=outpath, coloc=2)

    no_coloc_clust_stats = calculate_statistics(filt_cluster_data=no_coloc_df)

    coloc_clust_stats = calculate_statistics(filt_cluster_data=coloc_df)

    save_statistics(no_coloc_clust_stats, out=outpath, coloc=1)

    save_statistics(coloc_clust_stats, out=outpath, coloc=2)

    compare_clust_size(data=no_coloc_analysed_filt, coloc_data=coloc_analysed_filt,
                       out=outpath)
    
    compare_clust_circularity(data=no_coloc_analysed_filt, coloc_data=coloc_analysed_filt,
                              out=outpath)
    
    compare_clust_density(data=no_coloc_analysed_filt, coloc_data=coloc_analysed_filt,
                          out=outpath)

if __name__ == '__main__':

    two_color_reg_accuracy()