import numpy as np
from scipy import stats
from plots import plot_boxplot
import itertools

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

def kruskal_wallis(statistic_data: list, statistic: str, outpath: str) -> None:

    """
    This function carries out the Kruskal-Wallis test for a particular cluster statistic
    across different time points.

    In: statistic_data---a list of a particular cluster statistic at different time points.
    statistic---the cluster statistic on which the test will be carried out (str)
    outpath---the output folder where results will be saved (str)

    Out: a .txt file with the H-value and p-value from the Kruskal-Wallis test.
    """
    H, p = stats.kruskal(*statistic_data)

    with open(outpath + '/kruskal_wallis_result_' + statistic + '.txt', 'w') as f:

        f.write('The test statistic is: ' + str(H) + '\n')
        f.write('The p-value is: ' + str(p) + '\n')

def compare_mannwhit_pairs(statistic_data: list, statistic: str, outpath: str) -> None:

    """
    This function carries out pairwise Mann-Whitney tests between all time points. 

    In: statistic_data---a list of numpy arrays of a particular statistic at different time points.
    Statistic---the cluster statistic on which the test will be carried out (str)
    Outpath---where the results will be saved (str).

    Out: p_values---an array containing the p-values from the Mann-Whitney tests of all
    pairs of time points.
    """

    all_pairs = list(itertools.combinations(statistic_data, 2))

    p_values = np.zeros(len(statistic_data), 1)

    for i in range(0, len(all_pairs)):

        time_series_pair = all_pairs[i]

        p = mann_whitney_utest(time_series_pair[0], time_series_pair[1],
                           statistic=statistic + ' ' + str(i + 1), out=outpath)
        
        p_values[i] = p
    
    return p_values

def correct_pvalues(p_values: 'np.ndarray[np.float64]', statistic: str, outpath: str) -> 'np.ndarray[np.float64]':

    """
    This function corrects for multiple testing using the Benjamini-Hochberg method.

    In: p_values---a numpy array with the p-values resulting from Mann-Whitney tests of
    all pairs of time points.
    statistic---the cluster statistic on which the test will be carried out (str)
    outpath---where the results will be saved (str)

    Out: corrected_pvalues---p-values adjusted for multiple testing (np array)
    """
    
    corrected_pvalues = stats.false_discovery_correct(p_values)

    with open(outpath + '/cor_pvalues_' + statistic + '.txt', 'w') as f:

        for p in corrected_pvalues:

            f.write(str(p) + '\n')
    
    return correct_pvalues

def compare_radii_time_series(radii_data: list['np.array'], outpath: str) -> None:

    """
    This function carries out the Kruskal-Walis test, compares all pairs of datasets using the Mann-Whitney test,
    and plots a boxplot of varying radii.
    """

    kruskal_wallis(statistic_data=radii_data, statistic='Radius (nm)', outpath=outpath)

    pvals = compare_mannwhit_pairs(statistic_data=radii_data, statistic='Radius (nm)', outpath=outpath)

    correct_pvalues(pvals, statistic='Radius (nm)', outpath=outpath)

    plot_boxplot(data=radii_data, statistic='Radius (nm)', out=outpath)