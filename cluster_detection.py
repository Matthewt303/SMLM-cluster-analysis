import ripleyk
import numpy as np
from file_io import user_input, load_locs, extract_xy, save_max_r
from plots import plot_ripley_h

def generate_radii(bounding_radius: float, increment: float) -> list[float]:

    """
    Generate a list of radii for cluster detection via Ripley functions.
    Bounding radius is the maximum radius while the increment defines the range 
    of radii for the functions.

    In: the maximum radius of a region of interest (float).
    The size of the increment, starting from r = 0 (float).

    Out: a list of numbers representing the increasing radii of a circle
    from a point (list of floats.)
    """

    radii = list(np.arange(0, bounding_radius, increment))

    return radii[1:]

def ripley_k_function(xy_data: 'np.ndarray[np.float64]', r: list[float], br: float) -> 'np.ndarray[np.float64]':

    """
    2D Ripley's K=function. Converts result to numpy array.

    In: xy_data---xy localisations from STORM (np array)
    r---list of radii to calculate ripley's K-values (list of floats)
    br---bounding radius, maximum radius to calc K-values (float)

    Out: array of K-values.
    """

    k = ripleyk.calculate_ripley(r, br, d1=xy_data[:, 0], d2=xy_data[:, 1])
    
    return np.array(k).reshape(len(k), 1)

def ripley_l_function(k_values: 'np.ndarray[np.float64]') -> 'np.ndarray[np.float64]':

    """
    2D Ripley's L-function, normalized such that the expected value is r.

    In: k_values---Ripley K-values (numpy array)

    Out: Ripley L-values (numpy array)

    """

    return np.sqrt(k_values / np.pi)

def ripley_h_function(l_values: 'np.ndarray[np.float64]', radii: list[float]) -> 'np.ndarray[np.float64]':

    """
    2D Ripley's H-function, normalized such that the expected value is 0.
    The radii are converted to a numpy array.

    In: l_values---Ripley L-values (numpy array)

    Out: Ripley H-values (numpy array)
    """

    return l_values - np.array(radii).reshape(len(radii), 1)

def calculate_rmax(h_values: 'np.ndarray[np.float64]', radii: list[float]) -> float:

    """
    Calculate the radius at which Ripley's H-function is at a maximum

    In: h_values---Ripley h-values (numpy array)
    radii---list of radii over which Ripley's K-function was calculated (np array)

    Out: radius at which the H-values are at a maximum (float)
    """

    return radii[h_values.argmax()]

def main():

    """
    Contains functions for Ripley K-function cluster classification.
    """

    print('Enter path to localisation file')
    path = user_input()

    print('Enter folder for things to be saved.')
    outpath = user_input()
    
    data = load_locs(path)

    xy = extract_xy(data)
    
    radii = generate_radii(bounding_radius=1500,
                           increment=10)
    
    k_values = ripley_k_function(xy, r=radii, br=1500)

    l_values = ripley_l_function(k_values=k_values)

    h_values = ripley_h_function(l_values=l_values, radii=radii)

    plot_ripley_h(h_values=h_values, radii=radii, out=outpath)

    rmax = calculate_rmax(h_values=h_values, radii=radii)

    save_max_r(outpath=outpath, max_r=rmax)