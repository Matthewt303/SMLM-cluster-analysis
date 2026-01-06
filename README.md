# Overview

This repository contains a series of command-line scripts for colocalisation analysis and cluster analysis of SMLM localisation data.

## Prerequisites

- Python >= 3.11 < 3.14
- A Python virtual environment 
- Localisation data as .csv files in ThunderSTORM format
- Localisation data of fluorescent beads acquired in two different channels. Used for channel registration.

## Installation

Create a virtual environment then run:

```
git clone https://github.com/Matthewt303/SMLM-cluster-analysis.git
cd SMLM-cluster-analysis
python3 -m pip install .
```

## Usage

Following installation, a series of command-line scripts should become available.

For channel registration and colocalisation analysis:

```
analyze-coloc.exe --beads_ch1_file /path/to/bead_locs_channel_1.csv --beads_ch2_file /path/to/bead_locs_channel_2.csv --ch1_loc_file /path/to/channel_1_locs.csv --ch2_loc_file /path/to/channel_2_locs.csv --max_radius 125 --radius_increment 25 --out_folder /path/to/where/things/are/saved
```

The max radius and radius increment determine the search area for colocalisation analysis.

Outputs:

- Localisation files with CBC values

### Cluster detection

Uses Ripley's K-function [1] to detect clusters in localisation data.

```
detect-clusters --loc_file /path/to/localisations.csv --out_folder /path/to/where/things/are/saved --bounding_radius 1000 --radius_increment 50 --n_channels 1
```

Bounding radius and radius increment determine the search area for Ripley's K-function. n_channels is to specify whether the data come from single-color STORM or two-color STORM.

Outputs:

- Plot of Ripley's H-function
- Radius at which Ripley's H-function is at its maximum 

### Cluster analysis

Uses HDBSCAN [2] to cluster localisation data.

```
analyze-clusters.exe --loc_file /path/to/localisations.csv --out_folder /path/to/where/things/are/saved --min_cluster_size 4 --n_channels 2
```

```min_cluster_size``` determines the minimum number of localisations for a cluster.

Output:

- Histogram of each cluster parameters
- Cluster parameters for each cluster as a .csv file
- Cluster statistics in a .txt file

#### Analysis of cluster data

Uses principal component analysis to analyse cluster characteristics.

```
pca-clusters --in_folder /path/to/cluster_stats --out_folder /path/to/where/things/are/saved --conditions cond1, cond2, cond3 --n_components 2
```

Here, the input folder should contain a series of .csv files, each corresponding to cluster parameters from a different experimental condition. The number of conditions must match the number of .csv files. Note, two or three principal components are recommended.

Output:

- PCA scatterplot
- PCA loadings as a .txt file
- Proportion of explained variance as a .txt file

### Variance of PCA

Plots the explained variance for different principal components.

```
pca-clusters --in_folder /path/to/cluster_stats --out_folder /path/to/where/things/are/saved --conditions cond1, cond2, cond3
```

Output:
- Plot of explained variance.

## Acknowledgements

The colocalisation analysis is based off the approach presented by Malkusch *et al.* 2012 [3].

[1](https://www.sciencedirect.com/science/article/pii/S0006349509010480). Kiskowski, M. A., Hancock, J. F. & Kenworthy, A. K. On the Use of Ripley’s
K-Function and Its Derivatives to Analyze Domain Size. *Biophysical Journal* **97**,
1095–1103 (2009)

[2](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14). *Density-Based Clustering Based on Hierarchical Density Estimates* in *The 17th Pacific-Asia Conference on Knowledge Discovery and Data Mining* (eds Pei, J., Tseng, V. S., Cao, L., Motoda, H. & Xu, G.) (Springer, Gold Coast, QLD, Australia, 2013), 160–172

[3](https://link.springer.com/article/10.1007/s00418-011-0880-5). Malkusch, S., Endesfelder, U., Mondry, J., Gelleri, M., Verveer, P. J. & Heile-
mann, M. Coordinate-based colocalization analysis of single-molecule localization
microscopy data. *Histochemistry and Cell Biology* **137**, 1–10 (2012)