import numpy as np
import smlm_clust_analysis.internals.pca_funcs as pc
import smlm_clust_analysis.internals.file_io as io
from smlm_clust_analysis.internals.plots import plot_var_ratio 

def plot_var():

    input = "C:/Users/mxq76232/Documents/PhD/Project_work/" \
    "STORM_two_colour_re_analysis_27oct25/all"
    
    conditions = ["5 minutes", "15 minutes", "30 minutes"]

    file_list = io.collate_clust_files(input)
    all_clust_data = io.combine_all_stats(file_list, conditions)

    clust_data_array = pc.extract_features_asarray(all_clust_data)

    norm_clust_data = pc.z_norm_cluster_features(clust_data_array)

    _, __, variance_ratio = pc.pca(norm_clust_data, 5)
    plot_var_ratio(variance_ratio, input)

def main():

    input = "C:/Users/mxq76232/Documents/PhD/Project_work/" \
    "STORM_two_colour_re_analysis_27oct25/all"
    
    conditions = ["5 minutes", "15 minutes", "30 minutes"]

    file_list = io.collate_clust_files(input)
    all_clust_data = io.combine_all_stats(file_list, conditions)

    clust_data_array = pc.extract_features_asarray(all_clust_data)

    norm_clust_data = pc.z_norm_cluster_features(clust_data_array)

    pca_data, loadings, variance_ratio = pc.pca(norm_clust_data)
    io.save_expl_var(variance_ratio, input)

    pca_data_df = pc.convert_to_df_2d(pca_data)

    all_data_df = pc.generate_final_df(pca_data_df, all_clust_data)

if __name__ == "__main__":
    plot_var()