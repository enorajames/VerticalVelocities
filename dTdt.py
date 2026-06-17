#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:32:03 2026

@author: ejames
"""


import xarray as xr
import pandas as pd
import numpy as np

# parameters
years = range(1993,2020)

# box 
lon_min, lon_max = -55, -30
lat_min, lat_max = 20, 35



#%%
# loading nc files
nc_file_OLIV3 = '/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/oliv3_glob_025_ekf_monthly_verlev/oliv3_glob_025_ekf_monthly_verlev_2012.nc'

ds = xr.open_dataset(nc_file_OLIV3)

# geographic mask
lon = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
mask = (
    (lon >= lon_min) & (lon <= lon_max) &
    (ds.latitude >= lat_min) & (ds.latitude <= lat_max)
)

#%% compute dTdt
paths = [
    f"/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/ARMOR3D/{y}/armor3D_monthly_t_s_mld_{y}.nc"
    for y in years
]

#%%

ds_all = xr.open_mfdataset(paths, combine="by_coords", engine="netcdf4")
T = ds_all["tam"]

#%% limit to a horizontal box (lon, lat)
T = T.where(mask, drop=True)
dTdt = T.differentiate("time", datetime_unit='ns') * 1e9 # differentiate gives nano seconds, so *1e9 to have °C/s 
dTdt = dTdt.rename("dTdt")

#%%
dTdt.to_netcdf('/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/temperature_material_derivative_global/dTdt_100_200m_global_1993_2019.nc')