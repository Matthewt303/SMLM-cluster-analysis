import numpy as np
from scipy import stats
from plots import plot_boxplot

def mann_whitney_utest(data1: 'np.ndarray[np.float64]', data2: 'np.ndarray[np.float64]', statistic: str, out: str) -> float:

    """
    This function carries out the Mann-Whitney U test, also known as the Wilcoxon summed rank test,
    on two datasets. The alternative hypothesis is two-sided.

    In: data1---dataset of an independent variable (np array)
    data2---dataset of second independent variable (np array)
    statistic---the parameter that will be compared (str).
    out---output folder where the result will be saved (str).

    Out:
    .txt file with the U-statistic and significance value (float)
    p---the p-value from the significance test (float)
    """

    U1, p = stats.mannwhitneyu(data1, data2)

    with open(out + '/mann_whitney_utest_' + statistic + '_result.txt', 'w') as f:

        f.write('U-statistic: ' + str(U1) + '\n')
        f.write('p-value: ' + str(p) + '\n')

    return p

def compare_clust_size(data: 'np.ndarray[np.float64]', coloc_data: 'np.ndarray[np.float64]', out: str):

    """
    This function extracts the cluster radii from colocalised molecules and non-colocalised
    molecules and plots it as a boxplot. A two-sided Mann-Whitney U test is also carried out
    to test for significant difference.

    In: data---cluster data from non-colocalised molecules (np array)
    coloc_data---cluster data from colocalised molecules (np array)
    out---output folder where things will be saved (str)

    Out: None but a boxplot comparing cluster radii will be saved to the specified folder, as
    well as the result of the Mann-Whitney U test.
    """

    no_loc_radii, loc_radii = data[:, 3], coloc_data[:, 3]

    radii_data = [no_loc_radii, loc_radii]

    plot_boxplot(radii_data, statistic='Radius (nm)', out=out)

    mann_whitney_utest(no_loc_radii, loc_radii, statistic='Radius (nm)', out=out)

def compare_clust_circularity(data: 'np.ndarray[np.float64]', coloc_data: 'np.ndarray[np.float64]', out: str):

    """
    This function extracts the cluster circularity from colocalised molecules and non-colocalised
    molecules and plots it as a boxplot. A two-sided Mann-Whitney U test is also carried out
    to test for significant difference.

    In: data---cluster data from non-colocalised molecules (np array)
    coloc_data---cluster data from colocalised molecules (np array)
    out---output folder where things will be saved (str)

    Out: None but a boxplot comparing cluster circularity will be saved to the specified folder, as
    well as the result of the Mann-Whitney U test.
    """

    no_loc_circ, loc_circ = data[:, 4], coloc_data[:, 4]

    radii_data = [no_loc_circ, loc_circ]

    plot_boxplot(radii_data, statistic='Circularity', out=out)

    mann_whitney_utest(no_loc_circ, loc_circ, statistic='Circularity', out=out)

def compare_clust_density(data: 'np.ndarray[np.float64]', coloc_data: 'np.ndarray[np.float64]', out: str):

    
    """
    This function extracts the cluster density from colocalised molecules and non-colocalised
    molecules and plots it as a boxplot. A two-sided Mann-Whitney U test is also carried out
    to test for significant difference.

    In: data---cluster data from non-colocalised molecules (np array)
    coloc_data---cluster data from colocalised molecules (np array)
    out---output folder where things will be saved (str)

    Out: None but a boxplot comparing cluster density will be saved to the specified folder, as
    well as the result of the Mann-Whitney U test.

    """
    no_loc_density, loc_density = data[:, -2], coloc_data[:, -2]

    density_data = [no_loc_density, loc_density]

    plot_boxplot(density_data, statistic='Density (n . um^-2)', out=out)

    mann_whitney_utest(no_loc_density, loc_density, statistic='Density (n . um^-2)', out=out)