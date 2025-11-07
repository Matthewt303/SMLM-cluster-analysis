import smlm_clust_analysis.internals.file_io as io
import smlm_clust_analysis.internals.two_colour_funcs as tc
from smlm_clust_analysis.internals.ripley_funcs import generate_radii
import time
import argparse
import os

def check_args(args: object) -> None:
    """
    Check user-specified arguments. Checks:
    1) Existence of input folders and output folder
    """

    arg_dict = vars(args)

    for arg in arg_dict.values():
        if not arg:
            raise TypeError("One or more required arguments are missing.")

    if not os.path.isfile(arg_dict["beads_ch1_file"]):
        raise FileNotFoundError("Bead channel 1 file does not exist")

    if not os.path.isfile(arg_dict["beads_ch2_file"]):
        raise FileNotFoundError("Bead channel 2 file does not exist")

    if not os.path.isfile(arg_dict["ch1_loc_file"]):
        raise FileNotFoundError("Channel 1 locs file does not exist.")

    if not os.path.isfile(arg_dict["ch2_loc_file"]):
        raise TypeError("Channel 2 locs file does not exist.")
    
    if arg_dict["max_radius"] <= 0:
        raise ValueError("Maximum radius cannot be less than or equal to zero.")
    
    if arg_dict["radius_increment"] <= 0:
        raise ValueError("Radius increment cannot be less than or equal to zero")
    
    if arg_dict["max_radius"] <= arg_dict["radius_increment"]:
        raise ValueError("Max radius cannot be less than or the same as"
        "the radius increment.")

    if not os.path.isdir(arg_dict["out_folder"]):
        raise TypeError("Output folder does not exist.")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--beads_ch1_file", type=str)
    parser.add_argument("--beads_ch2_file", type=str)
    parser.add_argument("--ch1_loc_file", type=str)
    parser.add_argument("--ch2_loc_file", type=str)
    parser.add_argument("--max_radius", type=float)
    parser.add_argument("--radius_increment", type=float)
    parser.add_argument("--out_folder", type=str)

    opt = parser.parse_args()

    check_args(opt)

    start = time.perf_counter()

    green_beads, red_beads = io.load_locs(path=opt.beads_ch1_file), io.load_locs(path=opt.beads_ch2_file)

    green_locs, red_locs = io.load_locs(path=opt.ch1_loc_file), io.load_locs(path=opt.ch2_loc_file)

    green_locs_xy = io.extract_xy(locs=green_locs)
    
    green_bead_xy, red_bead_xy = io.extract_xy(locs=green_beads), io.extract_xy(locs=red_beads)

    matrix = tc.calculate_transformation_matrix(channel1=green_bead_xy, channel2=red_bead_xy)

    green_bead_xy_reg = tc.register_channel(channel=green_bead_xy, matrix=matrix)

    nearest_neighbors = tc.calculate_nneighbor_dist(ch1_locs=green_bead_xy_reg, ch2_locs=red_bead_xy, radii=[1, 1])

    green_xy_filt, red_xy_filt = tc.filter_bead_locs(ch1_locs=green_bead_xy, ch2_locs=red_bead_xy, nneighbors=nearest_neighbors)

    matrix_2 = tc.calculate_transformation_matrix(channel1=green_xy_filt, channel2=red_xy_filt)
    nneighbors, reg_error = tc.calc_reg_error(green_xy_filt, red_xy_filt, matrix_2)
    io.save_reg_error(opt.out_folder, reg_error)

    green_xy_reg = tc.register_channel(channel=green_locs_xy, matrix=matrix_2)

    green_locs_cor = io.save_corrected_channels(cor_locs=green_xy_reg, locs=green_locs, out=opt.out_folder)

    green, red = tc.add_channel(locs=green_locs_cor, channel=1), tc.add_channel(locs=red_locs, channel=2)

    green_xy, red_xy = io.extract_xy(green), io.extract_xy(red)
    
    radii = generate_radii(bounding_radius=opt.max_radius, increment=opt.radius_increment)

    areas = tc.convert_radii_to_areas(radii)

    gg_dist, gr_dist = tc.calc_all_distributions(green_xy, red_xy, radii, areas)
    
    green_spearman = tc.calc_spearman_cor_coeff(gg_dist, gr_dist)

    colocs = tc.calc_coloc_values(green_spearman, green_xy, red_xy, radii)

    io.save_locs_colocs(tc.add_coloc_values(locs=green, coloc_values=colocs),
                       channel=1, out=opt.out_folder)
    
    rr_dist, rg_dist = tc.calc_all_distributions(red_xy, green_xy, radii, areas)
    
    red_spearman = tc.calc_spearman_cor_coeff(rr_dist, rg_dist)

    colocs_red = tc.calc_coloc_values(red_spearman, red_xy, green_xy, radii)
    
    io.save_locs_colocs(tc.add_coloc_values(locs=red, coloc_values=colocs_red),
                     channel=2, out=opt.out_folder)
    
    all_locs = tc.combine_channel_locs(tc.add_coloc_values(locs=green, coloc_values=colocs),
                                    tc.add_coloc_values(locs=red, coloc_values=colocs_red))

    io.save_locs_colocs(all_locs, channel='all', out=opt.out_folder)

    end = time.perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))

if __name__ == "__main__":
    main()