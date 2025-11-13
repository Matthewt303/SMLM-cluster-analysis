import argparse
import warnings
import os 
import smlm_clust_analysis.internals.pca_funcs as pc
import smlm_clust_analysis.internals.file_io as io
from smlm_clust_analysis.internals.plots import plot_var_ratio, plot_components_2d, plot_components_3d

def check_var_args(args: object) -> None:

    arg_dict = vars(args)

    for arg in arg_dict.values():
        if not arg:
            raise TypeError("One or more required arguments are missing.")
    
    if not os.path.isdir(arg_dict["in_folder"]):
        raise TypeError("Input folder does not exist.")
    
    if not os.path.isdir(arg_dict["out_folder"]):
        raise TypeError("Output folder does not exist")
    
    if not isinstance(arg_dict["conditions"], list):
        raise TypeError("Please ensure the conditions are given as a list.")
    
    if len(arg_dict["conditions"]) < 1:
        raise ValueError("List of conditions cannot be empty.")
    
    if len(arg_dict["conditions"]) < 2:
        warnings.warn("Only one condition has been provided. The results of" \
        "PCA are unlikely to make much sense.")

def check_args(args: object) -> None:

    arg_dict = vars(args)

    for arg in arg_dict.values():
        if not arg:
            raise TypeError("One or more arguments are missing.")
    
    if not os.path.isdir(arg_dict["in_folder"]):
        raise TypeError("Input folder does not exist.")
    
    if not os.path.isdir(arg_dict["out_folder"]):
        raise TypeError("Output folder does not exist")
    
    if not isinstance(arg_dict["conditions"], list):
        raise TypeError("Please ensure the conditions are given as a list.")
    
    if len(arg_dict["conditions"]) < 1:
        raise ValueError("List of conditions cannot be empty.")
    
    if len(arg_dict["conditions"]) < 2:
        warnings.warn("Only one condition has been provided. The results of" \
        "PCA are unlikely to make much sense.")
    
    if arg_dict["n_components"] < 2 or arg_dict["n_components"] > 3:
        raise ValueError("Only two or three principal components are supported.")



def plot_var():

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_folder", type=str)
    parser.add_argument("--out_folder", type=str)
    parser.add_argument("--conditions", nargs="+")

    opt = parser.parse_args()
    check_var_args(opt)

    file_list = io.collate_clust_files(opt.in_folder)
    all_clust_data = io.combine_all_stats(file_list, opt.conditions)

    clust_data_array = pc.extract_features_asarray(all_clust_data)

    norm_clust_data = pc.z_norm_cluster_features(clust_data_array)

    _, __, variance_ratio = pc.pca(norm_clust_data, 5)
    plot_var_ratio(variance_ratio, opt.out_folder)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_folder", type=str)
    parser.add_argument("--out_folder", type=str)
    parser.add_argument("--conditions", nargs='+')
    parser.add_argument("--n_components", type=int)

    opt = parser.parse_args()
    check_args(opt)

    file_list = io.collate_clust_files(opt.in_folder)
    all_clust_data = io.combine_all_stats(file_list, opt.conditions)

    clust_data_array = pc.extract_features_asarray(all_clust_data)

    norm_clust_data = pc.z_norm_cluster_features(clust_data_array)

    pca_data, loadings, variance_ratio = pc.pca(norm_clust_data, opt.n_components)
    io.save_loadings(loadings, opt.out_folder)
    io.save_expl_var(variance_ratio, opt.out_folder)

    if opt.n_components == 2:

        pca_data_df = pc.convert_to_df_2d(pca_data)

        all_data_df = pc.generate_final_df(pca_data_df, all_clust_data)
        plot_components_2d(all_data_df, opt.out_folder)
    
    elif opt.n_components == 3:

        pca_data_df = pc.convert_to_df_3d(pca_data)

        all_data_df = pc.generate_final_df(pca_data_df, all_clust_data)
        plot_components_3d(all_data_df, opt.out_folder)
