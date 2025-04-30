import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA


def load_h5data(input_file):
    """Load dataset from an HDF5 file."""
    data = xr.open_dataset(input_file, engine="h5netcdf")
    return data

def combine_updn_data(ds: xr.Dataset) -> np.ndarray:
    """
    Flatten and combine 'up' and 'down' spin DataArrays from an xarray.Dataset.
    Args:
        ds: xarray.Dataset containing two DataArrays named up_var and dn_var.
            Each must have dimensions.
    Returns:
        A 2D NumPy array of shape
            (n_compositions, 2 * n_points),
    """
    # Extract raw NumPy arrays
    up = ds.up.values  # shape (n_comp, dim1, dim2)
    dn = ds.dn.values  # same shape

    # Ensure the composition dimension is first
    if up.ndim < 3 or dn.ndim < 3:
        raise ValueError(
            f"Expected at least 3D arrays for 'up' and 'dn', "
            f"got shapes {up.shape} and {dn.shape}"
        )

    # Flatten the last two dims
    n_comp = up.shape[0]
    n_points = up.shape[1] * up.shape[2]
    up_flat = up.reshape(n_comp, n_points)
    dn_flat = dn.reshape(n_comp, n_points)

    # Combine: up followed by negated down
    combined = np.hstack([up_flat, -dn_flat])

    return combined


def perform_pca(img_data, n_components=10):
    """Perform PCA on the image data."""
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(img_data)
    return pca, pca_result

def detect_outliers(pca_result, threshold):
    """Detect outliers based on jump differences in PC2."""
    score = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    jump_list = np.abs(np.diff(score["PC2"]))
    indices_above_threshold = np.where(jump_list > threshold)[0]
    return score, jump_list, indices_above_threshold

def plot_pca_contributions(pca):
    """Plot PCA contribution ratios and cumulative contributions."""
    contribution_ratios = pd.DataFrame(pca.explained_variance_ratio_)
    cumulative_contribution_ratios = contribution_ratios.cumsum()
    
    x_axis = range(1, len(contribution_ratios) + 1)
    plt.figure(figsize=(8, 6))
    plt.bar(x_axis, contribution_ratios.iloc[:, 0], align='center', label='Contribution Ratio')
    plt.plot(x_axis, cumulative_contribution_ratios.iloc[:, 0], 'r.-', label='Cumulative Ratio')
    plt.xlabel('Principal Components')
    plt.ylabel('Contribution Ratio')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

def plot_jump_list_vs_threshold(jump_list, threshold):
    """Plot jump values with threshold line."""
    number_list = np.arange(len(jump_list))
    plt.figure(figsize=(6, 6))
    plt.plot(number_list, jump_list, c="r", label="Jump values")
    plt.axhline(y=threshold, color="k", linestyle="--", label=f"y = {threshold}", lw=2)
    plt.xlabel('Index')
    plt.ylabel('Jump Value')
    plt.legend()
    plt.show()

def plot_spin_vs_composition(data, indices_above_threshold):
    """Plot spin polarization vs Ga composition."""
    my_list = np.linspace(0, 1, 101)
    plt.figure(figsize=(9, 6))
    plt.scatter(my_list, data["spin_polarization"], c=my_list, edgecolors="gray", cmap='winter', s=120)
    for idx in indices_above_threshold:
        plt.scatter(my_list[idx + 1], data["spin_polarization"].values[idx + 1], c="r", s=120)
    plt.show()

def plot_results(score, jump_list, indices_above_threshold, data, threshold, pca):
    """Generate various plots to visualize the PCA results and detected outliers."""
    # Scatter plot
    plt.figure(figsize=(6,6))
    for idx in indices_above_threshold:
        plt.scatter(-score['PC1'][idx], score['PC2'][idx], c="white", s=130, edgecolor='red', lw=2)
    plt.scatter(-score['PC1'], score['PC2'],  c=data["spin_polarization"].values, cmap='cool', edgecolor='gray', s=50)
    # plt.scatter(-score['PC1'], score['PC2'], c=data["spin_polarization"], cmap='cool', edgecolor='gray', s=50)
    
    for idx in indices_above_threshold:
        plt.plot(-score['PC1'][idx:idx+2], score['PC2'][idx:idx+2], lw=2, color="r", linestyle=":")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    # Histogram
    plt.hist(jump_list, bins=30, edgecolor='black')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')
    plt.show()
    
    # Additional Plots
    plot_pca_contributions(pca)
    plot_jump_list_vs_threshold(jump_list, threshold)
    plot_spin_vs_composition(data, indices_above_threshold)
