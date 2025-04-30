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

## Repository structure
. ├── README.md 
  ├── LICENSE
  ├── config.json # JSON parameters (paths, sigma, pixel size, PCA threshold…) 
  ├── data_processor.py # Preprocessing, mapping, saving, and Gaussian routines 
  ├── pca_processor.py # PCA, outlier detection, and plotting utilities 
  ├── main.ipynb # Jupyter notebook: full pipeline example 
  ├── DOS_editter.ipynb # Jupyter notebook: DOS editing & visualization 
  └── data/ 
       ├── fermi_line_kx_ky_result_up/ 
       ├── fermi_line_kx_ky_result_down/ 
       ├── gaussian/… └── h5/…
