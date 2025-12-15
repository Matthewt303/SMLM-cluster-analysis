import argparse
import os
import smlm_clust_analysis.internals.file_io as io
import smlm_clust_analysis.internals.ripley_funcs as rf
from smlm_clust_analysis.internals.plots import plot_ripley_h

def check_args(args: object) -> None:
    """
    Check user-specified arguments. Checks:
    1) Existence of input folders and output folder
    2) Numerical validity of radius and radius increment.
    """

    arg_dict = vars(args)

    for arg in arg_dict.values():
        if not arg:
            raise TypeError("One or more required arguments are missing.")

    if not os.path.isfile(arg_dict["loc_file"]):
        raise FileNotFoundError("Loc file does not exist")

    if not os.path.isdir(arg_dict["out_folder"]):
        raise TypeError("Output folder does not exist.")
    
    if arg_dict["bounding_radius"] <= 0:
        raise ValueError("Bounding radius cannot be below zero.")
    
    if arg_dict["radius_increment"] <= 0:
        raise ValueError("Radius increment cannot be below zero.")
    
    if arg_dict["radius_increment"] > arg_dict["bounding_radius"]:
        raise ValueError("Radius increment cannot be larger than the radius.")
    
    if arg_dict["n_channels"] < 1:
        raise ValueError("Number of channels cannot be zero")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--loc_file", type=str)
    parser.add_argument("--out_folder", type=str)
    parser.add_argument("--bounding_radius", type=int)
    parser.add_argument("--radius_increment", type=int)
    parser.add_argument("--n_channels", type=int)

    opt = parser.parse_args()
    check_args(opt)
    
    data = io.load_locs(opt.loc_file, channels=opt.n_channels)

    xy = io.extract_xy(data)
    
    radii = rf.generate_radii(bounding_radius=opt.bounding_radius,
                           increment=opt.radius_increment)
    
    k_values = rf.ripley_k_function(xy, r=radii, br=1500)

    l_values = rf.ripley_l_function(k_values=k_values)

    h_values = rf.ripley_h_function(l_values=l_values, radii=radii)

    plot_ripley_h(h_values=h_values, radii=radii, out=opt.out_folder)

    rmax = rf.calculate_rmax(h_values=h_values, radii=radii)

    io.save_max_r(outpath=opt.out_folder, max_r=rmax)

if __name__ == "__main__":
    main()