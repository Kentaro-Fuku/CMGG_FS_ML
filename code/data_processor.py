# Affin translation
# blurring

import os
import numpy as np
import pandas as pd
import xarray as xr
import math
import json
from typing import Dict, Any
from scipy.ndimage import gaussian_filter
import h5netcdf

def load_config(json_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    Args:
        json_path: Path to the JSON configuration file.
    Returns:
        A dict containing all configuration parameters.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class Preprocessor():
    
    def __init__(self, cfg: Dict):
        self.base = cfg["BASE_PATH"]
        self.inp = cfg["INPUT_PATHS"]
        self.bz = cfg["BZ_FILE_PATH"]
        self.angle = cfg["ROTATION_ANGLE"]
        self._calc_translation()
        self.range_pixel = (self.T/2*5)/math.sqrt(2)
        self.pixel_size = cfg["PIXEL_SIZE"]
        self.sigma = cfg["SIGMA"]
        self.spin_csv = cfg["SPIN_P_CSV"]
        self._load_spin_composition()
        
        
    def _calc_translation(self):
        bz_data = np.loadtxt(os.path.join(self.base, self.bz))
        self.T = float(np.max(np.abs(bz_data)))
        self.trans_x = [
            np.array([ 2*self.T,        0]),
            np.array([-2*self.T,        0]),
            np.array([       0,  2*self.T]),
            np.array([       0, -2*self.T])]
        self.trans_g = [
            np.array([  self.T,    self.T]),
            np.array([ -self.T,    self.T]),
            np.array([  self.T,   -self.T]),
            np.array([ -self.T,   -self.T])]
        return None
    
    def _load_spin_composition(self):
        df = pd.read_csv(os.path.join(self.base, self.spin_csv))
        self.spin_p = df["spin_polalization"]
        self.composition = df["composition"]
        return None
        
    
    def _load_dat(self, path: str) -> np.ndarray:
        try:
            d = np.loadtxt(path)
            return d if d.size else np.empty((0, 2))
        except FileNotFoundError:
            print("files are not found.")
            return np.empty((0, 2))
        except Exception:
            print("error!")
            return np.empty((0, 2))
        
        
    
    def _rotation(self, data: np.ndarray) ->np.ndarray:
        θ = np.deg2rad(self.angle)
        R = np.array([[np.cos(θ), -np.sin(θ)],
                      [np.sin(θ), np.cos(θ)]])
        return np.dot(data, R)
        
        
    def _transform(self, data: dict) -> np.ndarray:
        up_tmp = np.empty((0, 2))
        dn_tmp = np.empty((0, 2))
        if data.get("up_x").size > 0:
            translated_up_x = np.vstack([data["up_x"] + t for t in self.trans_x] if data["up_x"].size else [])
            up_tmp = np.vstack([up_tmp] + [data["up_x"]] + [translated_up_x]) if data["up_x"].size else np.empty((0, 2))
        
        if data.get("up_g").size > 0:
            translated_up_g = np.vstack([data["up_g"] + t for t in self.trans_g] if data["up_g"].size else [])
            up_tmp = np.vstack([up_tmp] + [translated_up_g]) if data["up_g"].size else np.empty((0, 2))
        
        if data.get("dn_g").size > 0:
            translated_dn_g = np.vstack([data["dn_g"] + t for t in self.trans_g] if data["dn_g"].size else [])
            dn_tmp = np.vstack([dn_tmp] + [translated_dn_g]) if data["dn_g"].size else np.empty((0, 2))
        
        transformed_data = {"up": self._rotation(up_tmp),
                            "dn": self._rotation(dn_tmp)}
        return transformed_data

    
    def _mapping_data_in_BZ(self, filename: str) -> xr.Dataset:
        name = filename.replace('.dat', '')
        up_x = self._load_dat(os.path.join(self.base, self.inp["up_x"], filename))
        up_g = self._load_dat(os.path.join(self.base, self.inp["up_ganma"], filename))
        dn_g = self._load_dat(os.path.join(self.base, self.inp["down_ganma"], filename))
        data_dict = {"up_x":up_x,
                     "up_g":up_g,
                     "dn_g":dn_g}
        # print(data_dict["up_x"].shape, data_dict["up_g"].shape)
        BZmapped_dat = self._transform(data_dict)
        
        return BZmapped_dat
        
    def _coordinates_to_pixels(self, data: dict) -> dict:
        """
        Convert each (x,y) array in `data` into pixel indices.
        Returns a dict mapping the same keys to arrays of shape (n,2) of ints.
        """
        coordinate_data = {}
        for key, arr in data.items():
            # 空データは空配列を返却
            if arr.size == 0:
                coordinate_data[key] = np.empty((0, 2), dtype=int).tolist()
                continue

            # normalize coords from [-range_pixel, +range_pixel] to [0,1]
            norm_x = (arr[:, 0] + self.range_pixel) / (2 * self.range_pixel)
            norm_y = (arr[:, 1] + self.range_pixel) / (2 * self.range_pixel)

            # scale to pixel grid [0, pixel_size)
            pixel_x = (norm_x * self.pixel_size).astype(int)
            pixel_y = (norm_y * self.pixel_size).astype(int)

            # stack into (n,2)
            coordinate_data[key] = np.column_stack((pixel_x, pixel_y)).tolist()

        return coordinate_data 


    def preprocess(self, save_json=False):
        dataset = {"spin_polarization":self.spin_p,
                   "composition":self.composition}
        
        all_coordinates_dict = {}
        for filename in os.listdir(os.path.join(self.base, next(iter(self.inp.values())))):
            mapped_2Ddata = self._mapping_data_in_BZ(filename)
            coordinates = self._coordinates_to_pixels(mapped_2Ddata)
            all_coordinates_dict[filename] = coordinates
        dataset["coordinates"] = all_coordinates_dict
        if save_json:
            with open(os.path.join(self.base, 'dataset.json'), 'w') as f:
                json.dump(dataset, f)
        return dataset

    def gaussian_gradient(self, white_pixel_positions, sigma=None):
        """
        Fast gaussian broadening:
        1) rasterize white_pixel_positions into a sparse image
        2) convolve with Gaussian via scipy.ndimage.gaussian_filter
        3) normalize to [0,1]

        Args:
            white_pixel_positions: array-like of shape (N,2), pixel coords (x,y),
                                or empty list/array
            sigma: float or None. If None, use self.sigma.
        Returns:
            2D numpy array of shape (pixel_size, pixel_size), dtype float32.
        """
        if sigma is None:
            sigma = self.sigma
        size = self.pixel_size

        # 1) Create sparse image
        img = np.zeros((size, size), dtype=float)

        # Convert to numpy array
        coords = np.asarray(white_pixel_positions, dtype=int)

        # Handle empty input
        if coords.size == 0:
            return img.astype(np.float32)

        # If a single point is passed as shape (2,), reshape to (1,2)
        if coords.ndim == 1:
            if coords.shape[0] == 2:
                coords = coords.reshape(1, 2)
            else:
                raise ValueError(
                    f"Expected white_pixel_positions of shape (N,2), got {coords.shape}"
                )

        # Now coords should be (N,2)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Expected white_pixel_positions of shape (N,2), got {coords.shape}"
            )

        # Clip coords into valid range
        mask = (
            (coords[:, 0] >= 0) & (coords[:, 0] < size) &
            (coords[:, 1] >= 0) & (coords[:, 1] < size)
        )
        coords = coords[mask]

        # Accumulate counts at each coordinate (row=y, col=x)
        np.add.at(img, (coords[:, 1], coords[:, 0]), 1)

        # 2) Fast Gaussian filter (C-optimized)
        img = gaussian_filter(img, sigma=sigma, mode='constant')

        # 3) Normalize to [0,1]
        minv, maxv = img.min(), img.max()
        if maxv > minv:
            img = (img - minv) / (maxv - minv)

        return img.astype(np.float32)


    def gaussian_broadening_seq(
        self,
        dataset: Dict[str, Dict[str, np.ndarray]],
        sigma: float = None,
        save_h5: bool = False
    ) -> xr.Dataset:
        """
        Apply gaussian broadening to a sequence of up/down pixel positions,
        stack results into an xarray.Dataset, and optionally save to NetCDF.
        """
        # 1) determine sigma
        if sigma is None:
            sigma = self.sigma
        ps = self.pixel_size
        
        data = dataset["coordinates"]
        
        # 2) prepare arrays
        keys = list(data.keys())
        n = len(keys)
        up_arr = np.empty((n, ps, ps), dtype=np.float32)
        dn_arr = np.empty((n, ps, ps), dtype=np.float32)

        # 3) fill arrays
        for i, key in enumerate(keys):
            up_arr[i] = self.gaussian_gradient(data[key]["up"], sigma=sigma)
            dn_arr[i] = self.gaussian_gradient(data[key]["dn"], sigma=sigma)

        # 4) build coords
        x_coords = np.arange(ps)  # 0,1,...,ps-1
        y_coords = np.arange(ps)

        # 5) create Dataset
        ds = xr.Dataset(
            data_vars={
                "up":   (("composition", "x", "y"), up_arr),
                "dn":   (("composition", "x", "y"), dn_arr),
                # filenames as a data variable (non-coordinate)
                "filenames": (("composition",), keys),
            },
            coords={
                "composition": self.composition,
                "x":                 x_coords,
                "y":                 y_coords,
                "spin_polarization": self.spin_p
            },
            attrs={
                "sigma": sigma,
                "description": "Gaussian-broadened up/down images",
            }
        )

        if save_h5:
            out_path = os.path.join(self.base, f"data_sigma{sigma}.h5")
            ds.to_netcdf(out_path, mode="w", engine="h5netcdf")

        return ds
