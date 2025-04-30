# CMGG-FS-ML

**A Python pipeline for SARPES data processing and PCA-based outlier detection**

Co₂MnGaGe spin-resolved ARPES (SARPES) data are processed through:

1. **Brillouin Zone mapping**  
   – Load `.dat` files, apply parallel translations & rotations  
2. **Data export**  
   – Save as NumPy, NetCDF/HDF5, and image formats  
3. **Gaussian broadening**  
   – Rasterize coordinates, apply fast Gaussian filter  
4. **PCA analysis**  
   – Flatten up/down spin images, compute PCA, detect outliers, and generate plots  

---  

## Requirements

- Python 3.8+
- NumPy
- SciPy
- pandas
- xarray
- h5netcdf
- matplotlib
- scikit-learn





