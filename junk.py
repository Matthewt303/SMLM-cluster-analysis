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

def two_color_analysis_all():

    """
    All functions for two-colour analysis. Loads bead localisations and
    protein localisations, and performs channel registration. Saves corrected
    localisations then calculates cbc for all localisations. This is done for both
    channels, and the cbc values are appended to the localisation table which
    is saved as a .csv. Both channels are also combined into one localisation table
    and saved.
    """

    start = time.perf_counter()

    print('Enter path to beads for green channel.')
    green_bead_ch_path = user_input()

    print('Enter path to beads for red channel.')
    red_bead_ch_path = user_input()

    print('Enter path to localisations in green channel.')
    green_ch_path = user_input()

    print('Enter path to localisations in red channel.')
    red_ch_path = user_input()

    print('Where you want things saved.')
    out = user_input()

    green_beads, red_beads = load_locs(path=green_bead_ch_path), load_locs(path=red_bead_ch_path)

    green_locs, red_locs = load_locs(path=green_ch_path), load_locs(path=red_ch_path)

    green_locs_xy = extract_xy_cr(locs=green_locs)
    
    green_bead_xy, red_bead_xy = extract_xy_cr(locs=green_beads), extract_xy_cr(locs=red_beads)

    matrix = calculate_transformation_matrix(channel1=green_bead_xy, channel2=red_bead_xy)

    green_bead_xy_reg = register_channel(channel=green_bead_xy, matrix=matrix)

    nearest_neighbors = calculate_nneighbor_dist(ch1_locs=green_bead_xy_reg, ch2_locs=red_bead_xy, radii=[1, 1])

    green_xy_filt, red_xy_filt = filter_bead_locs(ch1_locs=green_bead_xy, ch2_locs=red_bead_xy, nneighbors=nearest_neighbors)

    matrix_2 = calculate_transformation_matrix(channel1=green_xy_filt, channel2=red_xy_filt)

    green_xy_reg = register_channel(channel=green_locs_xy, matrix=matrix_2)

    green_locs_cor = save_corrected_channels(cor_locs=green_xy_reg, locs=green_locs, out=out)

    green, red = add_channel(locs=green_locs_cor, channel=1), add_channel(locs=red_locs, channel=2)

    green_xy, red_xy = extract_xy(green), extract_xy(red)
    
    radii = generate_radii(bounding_radius=125, increment=25)

    areas = convert_radii_to_areas(radii)

    gg_dist, gr_dist = calc_all_distributions(green_xy, red_xy, radii, areas)
    
    green_spearman = calc_spearman_cor_coeff(gg_dist, gr_dist)

    colocs = calc_coloc_values(green_spearman, green_xy, red_xy, radii)

    save_locs_colocs(add_coloc_values(locs=green, coloc_values=colocs),
                       channel=1, out=out)
    
    rr_dist, rg_dist = calc_all_distributions(red_xy, green_xy, radii, areas)
    
    red_spearman = calc_spearman_cor_coeff(rr_dist, rg_dist)

    colocs_red = calc_coloc_values(red_spearman, red_xy, green_xy, radii)
    
    save_locs_colocs(add_coloc_values(locs=red, coloc_values=colocs_red),
                     channel=2, out=out)
    
    all_locs = combine_channel_locs(add_coloc_values(locs=green, coloc_values=colocs),
                                    add_coloc_values(locs=red, coloc_values=colocs_red))

    save_locs_colocs(all_locs, channel=3, out=out)

    end = time.perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))

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

def cluster_class_coloc_vs_no_coloc():

    """
    This function also calculates statistics and plots them following separation
    into colocalised vs non-colocalised. However, the separation is carried out
    prior to HDBSCAN.
    """

    print('Enter file path to localisation data: ')
    path = user_input()

    print('Enter folder where you want things stored: ')
    outpath = user_input()

    data = load_locs(path, channels=2)

    nocoloc, coloc = separate_coloc_data(data)

    clusters_nocoloc, clusters_coloc = hdbscan(nocoloc, min_n=4), hdbscan(coloc, min_n=4)

    clusters_nocoloc_filt, clusters_coloc_filt = denoise_data(clusters_nocoloc, min_n=4), denoise_data(clusters_coloc, min_n=4)
    
    save_dbscan_results(clusters_nocoloc_filt, n_channels=2, outpath=outpath)

    save_dbscan_results(clusters_coloc_filt, n_channels=2, outpath=outpath, filt=1)

    no_coloc_analysed, coloc_analysed = analyse_clusters(dbscan_data=clusters_nocoloc_filt), analyse_clusters(dbscan_data=clusters_coloc_filt)

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