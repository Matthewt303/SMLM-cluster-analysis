import argparse
import os
import smlm_clust_analysis.internals.cluster_funcs as clst
from smlm_clust_analysis.internals.file_io import load_locs, save_dbscan_results
from smlm_clust_analysis.internals.file_io import save_cluster_analysis, save_statistics
from smlm_clust_analysis.internals.plots import plot_cluster_statistics


def check_args(args: object) -> None:
    """
    Check user-specified arguments. Checks:
    1) Existence of input folders and output folder
    2) Numerical validity of cluster size.
    """

    arg_dict = vars(args)

    for arg in arg_dict.values():
        if not arg:
            raise TypeError("One or more required arguments are missing.")

    if not os.path.isfile(arg_dict["loc_file"]):
        raise FileNotFoundError("Loc file does not exist")

    if not os.path.isdir(arg_dict["out_folder"]):
        raise TypeError("Output folder does not exist.")

    if arg_dict["min_cluster_size"] <= 1:
        raise ValueError("Cluster size cannot be one or below.")


def main():
    """
    This function also calculates statistics and plots them following separation
    into colocalised vs non-colocalised. However, the separation is carried out
    prior to HDBSCAN.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--loc_file", type=str)
    parser.add_argument("--out_folder", type=str)
    parser.add_argument("--min_cluster_size", type=int)

    opt = parser.parse_args()
    check_args(opt)

    data = load_locs(opt.loc_file, channels=2)

    _, coloc = clst.separate_coloc_data(data)

    clusters_coloc = clst.hdbscan(coloc, min_n=opt.min_cluser_size)

    clusters_coloc_filt = clst.denoise_data(clusters_coloc, min_n=opt.min_cluser_size)

    save_dbscan_results(
        clusters_coloc_filt, n_channels=2, outpath=opt.out_folder, filt=1
    )

    coloc_analysed = clst.analyse_clusters(dbscan_data=clusters_coloc_filt)

    coloc_analysed_filt = clst.filter_clusters(coloc_analysed)

    coloc_df = clst.convert_to_dataframe(coloc_analysed_filt)

    save_cluster_analysis(filt_cluster_data=coloc_df, outpath=opt.out_folder)

    plot_cluster_statistics(filt_cluster_data=coloc_df, outpath=opt.out_folder, coloc=2)

    coloc_clust_stats = clst.calculate_statistics(filt_cluster_data=coloc_df)

    save_statistics(coloc_clust_stats, out=opt.out_folder, coloc=2)


if __name__ == "__main__":
    main()
