import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def extract_features_asarray(cluster_data: "pd.DataFrame") -> 'np.ndarray':
    
    clust_features = cluster_data[cluster_data.columns[:-1]]
    
    return np.array(clust_features).astype(np.float32)

def z_normalize(data: 'np.ndarray') -> 'np.ndarray':
    
    return (data - np.mean(data)) / np.std(data)

def z_norm_cluster_features(cluster_features: 'np.ndarray') -> 'np.ndarray':
    
    norm_data = np.zeros((cluster_features.shape[0],
                          cluster_features.shape[1]),
                         dtype=np.float32)
    
    for i, feature in enumerate(cluster_features.T):
        
        norm_data[:, i] = z_normalize(feature)
    
    return norm_data

def pca(norm_features: 'np.ndarray') -> tuple['np.ndarray']:

    """
    Carries out principal component analysis on normalized cluster features.
    The function reduces the dimensionality of the cluster data and returns the
    loadings of the first two principal components.
    """
    
    pca = PCA(n_components=2)
    
    data_pca = pca.fit_transform(norm_features)
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return data_pca, loadings

def convert_to_df_2d(principal_components: 'np.ndarray') -> 'pd.DataFrame':
    
    if principal_components.shape[1] != 2:
        
        raise IndexError('Mismatch between number of principal components and'
                         'final dimensions.')
    
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['PC1 Reduced Data', 'PC2 Reduced Data'])
    
    return principal_df

def convert_to_df_3d(principal_components: 'np.ndarray') -> 'pd.DataFrame':
    
    if principal_components.shape[1] != 3:
        
        raise IndexError('Mismatch between number of principal components and'
                         'final dimensions.')
    
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['Principal component 1',
                                         'Principal component 2',
                                         'Principal component 3'
                                    ])
    
    return principal_df

def generate_final_df(reduced_df: 'pd.DataFrame', original_df: 'pd.DataFrame') -> 'pd.DataFrame':
    
    final_df = pd.concat([reduced_df, original_df], axis=1)
    
    return final_df

    
def plot_components_2d(final_df: 'pd.DataFrame') -> None:
    
    fig, ax = plt.subplots()
    
    conditions = set(final_df[final_df.columns[-1]])
    
    colors = iter(plt.cm.viridis(np.linspace(0, 1, final_df.shape[1])))
    
    for condition in conditions:
        
        indices = final_df[final_df.columns[-1]] == condition
        
        ax.scatter(final_df.loc[indices, 'PC1 Reduced Data'], 
                   final_df.loc[indices, 'PC2 Reduced Data'],
                   c=next(colors), s=80, label=condition)
        

def plot_components_3d(final_df: 'pd.DataFrame'):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    targets = set(final_df[final_df.columns[-1]])
    
    colors = iter(plt.cm.viridis(np.linspace(0, 1, final_df.shape[1])))
    
    for target in targets:
        
        indices = final_df[final_df.columns[-1]] == target
    
        ax.scatter(final_df.loc[indices, 'Principal component 1'],
                   final_df.loc[indices, 'Principal component 2'],
                   final_df.loc[indices, 'Principal component 3'],
                   c=next(colors), s=60, label=target)
    
