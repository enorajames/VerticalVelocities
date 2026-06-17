#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:30:46 2026

@author: ejames
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd


# parameters
years = range(1993, 2020)

# box 
lon_min, lon_max = -55, -30
lat_min, lat_max = 20, 35

depth_min = 17  # 100 to 200 m
depth_max = 21


# loading nc files
nc_file_OLIV3 = '/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/oliv3_glob_025_ekf_monthly_verlev/oliv3_glob_025_ekf_monthly_verlev_2013.nc'

ds = xr.open_dataset(nc_file_OLIV3)

# geographic mask
lon = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
mask = (
    (lon >= lon_min) & (lon <= lon_max) &
    (ds.latitude >= lat_min) & (ds.latitude <= lat_max)
)


#%%

# compute annual average horizontal advection for each file
all_years = []

for year in years:
  
    # loading temperature file
    file = f'/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/ARMOR3D/{year}/armor3d_monthly_tsmld_{year}.nc'
    ds_A_stm = xr.open_dataset(file)
    T = ds_A_stm["tam"]

    T = T.assign_coords(deptht=(('deptht',), ds["depth"].values)) # we have to do this because dz is not constant

    # apply geographic mask
    T = T.where(mask, drop=True)

    # compute dTdz
    # '-' because we want shallow minus deep T but here differentiate does deep minus shallow
    # on deptht which now has the values of the depth and not from 0 to 49
    dTdz = -T.differentiate("deptht") 
    
    # select depth slice
    dTdz = dTdz.isel(deptht=slice(depth_min, depth_max + 1))
    
    print(dTdz)

    # loading w_oliv3 data
    file_OLIV3 = f'/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/oliv3_glob_025_ekf_monthly_verlev/oliv3_glob_025_ekf_monthly_verlev_{year}.nc'
    ds_O = xr.open_dataset(file_OLIV3)
    w = ds_O['w_oliv3']
    w = w.where(mask, drop=True)
    w = w.isel(deptht=slice(depth_min, depth_max + 1)).squeeze()
    
    # compute vertical advection anuual mean or monthly mean
    adv_v = w*dTdz
    
    # averaging over space / time / depth
    # adv_v_mean = adv_v.mean(dim=["time", "y", "x"])
    # adv_v_mean = adv_v.mean(dim=["y", "x"])

    # depth_vals = ds['depth'].isel(deptht=slice(depth_min, depth_max+1))
    # dz = depth_vals.diff("deptht")
    # dz = xr.concat([dz, dz.isel(deptht=-1)], dim="deptht")
    # adv_v_mean = (adv_v * dz).sum("deptht") / dz.sum()
    # adv_v_mean = adv_v_mean.mean(dim=["y", "x"])
    
    adv_v_mean= adv_v.mean('deptht')
    print(adv_v_mean)
    
    # add year dimension
    adv_v_mean = adv_v_mean.expand_dims("year")
    adv_v_mean = adv_v_mean.assign_coords(year=[year])
    
    all_years.append(adv_v_mean)
    
    print(year)

# concat on a year dimension
adv_v_all = xr.concat(all_years, dim="year")
print(adv_v_all)

#%%
adv_v_all = adv_v_all.stack(time_new=("year", "time"))
dates = pd.to_datetime(
    [f"{year}-{int(month):02d}-01" for year, month in zip(adv_v_all['year'].values, adv_v_all['time'].values)],
    format='%Y-%m-%d'
)
adv_v_all = adv_v_all.drop_vars(['time_new', 'year', 'time'])
adv_v_all.coords['time_new'] = ("time_new", dates)
#%%
adv_v_all.to_netcdf('/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/temperature_material_derivative_global/adv_v_100_200m_global_1993_2019.nc')