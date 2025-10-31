import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def extract_cluster_features(cluster_data: 'np.ndarray') -> 'np.ndarray':
    
    return cluster_data[:, 2:-1]

def z_normalize(data: 'np.ndarray') -> 'np.ndarray':
    
    return (data - np.mean(data)) / np.std(data)

def z_norm_cluster_features(cluster_features: 'np.ndarray') -> 'np.ndarray':
    
    norm_data = np.zeros((cluster_features.shape[0],
                          cluster_features.shape[1]),
                         dtype=np.float32)
    
    for i, feature in enumerate(cluster_features.T):
        
        norm_data[:, i] = z_normalize(feature)
    
    return norm_data

def pca(norm_features: 'np.ndarray') -> 'np.ndarray':
    
    pca = PCA(n_components=3)
    
    principalComponents = pca.fit_transform(norm_features)
    
    #loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return principalComponents

def convert_to_df_2d(principal_components: 'np.ndarray') -> 'pd.DataFrame':
    
    if principal_components.shape[1] != 2:
        
        raise IndexError('Mismatch between number of principal components and'
                         'final dimensions.')
    
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['Principal component 1', 'Principal component 2'])
    
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

def generate_final_df(principal_df: 'pd.DataFrame', original_df: 'pd.DataFrame') -> 'pd.DataFrame':
    
    final_df = pd.concat([principal_df, original_df[original_df.columns[-3:]]])
    
    return final_df

    
def plot_components_2d(final_df: 'pd.DataFrame') -> None:
    
    fig, ax = plt.subplots()
    
    targets = set(final_df[final_df.columns[-1]])
    
    colors = iter(plt.cm.viridis(np.linspace(0, 1, final_df.shape[1])))
    
    for target in targets:
        
        indices = final_df[final_df.columns[-1]] == target
        
        ax.scatter(final_df.loc[indices, 'Principal component 1'], 
                   final_df.loc[indices, 'Principal component 2'],
                   c=next(colors), s=80, label=target)
        

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
    
