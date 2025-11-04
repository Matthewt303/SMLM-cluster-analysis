import numpy as np
import smlm_clust_analysis.internals.pca_funcs as pc
import smlm_clust_analysis.internals.file_io as io

def main():

    input = "C:/Users/mxq76232/Documents/PhD/Project_work/" \
    "STORM_two_colour_re_analysis_27oct25/all"
    
    conditions = ["5 minutes", "15 minutes", "30 minutes"]

    file_list = io.collate_clust_files(input)
    all_clust_data = io.combine_all_stats(file_list, conditions)

    clust_data_array = pc.extract_features_asarray(all_clust_data)

    norm_clust_data = pc.z_norm_cluster_features(clust_data_array)

    print(norm_clust_data[0:6, :])
    print(np.mean(norm_clust_data, axis=0))

if __name__ == "__main__":
    main()