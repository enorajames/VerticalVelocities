#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:56:11 2026

@author: ejames
"""

import numpy as np
import xarray as xr



# parameters
years = range(1993, 2020)

# box 
lon_min, lon_max = -55, -30
lat_min, lat_max = 20, 35

depth_min = 17 # put the index of minimal chosen depth
depth_max = 21

R = 6.371e6                             # Earth radius
dlat_rad = np.deg2rad(0.25)             # Grid resolution lat
dlon_rad = np.deg2rad(0.25)             # Grid resolution lon

# loading nc files
nc_file_OLIV3 = '/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/oliv3_glob_025_ekf_monthly_verlev/oliv3_glob_025_ekf_monthly_verlev_2012.nc'
ds = xr.open_dataset(nc_file_OLIV3)

# geographic mask
lon = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
mask = (
    (lon >= lon_min) & (lon <= lon_max) &
    (ds.latitude >= lat_min) & (ds.latitude <= lat_max)
)
#%%

# compute horizontal advection for each file
all_years = []

for year in years:
  
    print('Start computing horizontal advection of ', year)
  
    # loading temperature file
    T_file = f'/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/ARMOR3D/{year}/armor3d_monthly_tsmld_{year}.nc'
    ds_A_stm = xr.open_dataset(T_file)
    
    T = ds_A_stm["tam"]

    # apply geographic mask
    # T = T.where(mask, drop=True)
    T = T.isel(deptht=slice(depth_min, depth_max + 1)) 
    
    # compute dT/dx and dT/dx
    dx = R * np.cos(np.deg2rad(T.latitude)) * dlon_rad
    dy = R * dlat_rad
    
    dTdx = T.differentiate("x") / dx
    dTdy = T.differentiate("y") / dy
        
    # geotrophic u v ARMOR3D file
    uv_file = f'/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/ARMOR3D/{year}/armor3d_monthly_uv_{year}.nc'
    ds_uv = xr.open_dataset(uv_file)
    
    u = ds_uv.uam.isel(deptht=slice(depth_min, depth_max + 1))
    v = ds_uv.vam.isel(deptht=slice(depth_min, depth_max + 1))
    
    # u = u.where(mask, drop=True)
    # v = v.where(mask, drop=True)
    
    # compute advection
    adv_h = u*dTdx + v*dTdy
    
    adv_h = adv_h.mean('deptht')
    
    # add year dimension
    adv_h = adv_h.expand_dims("year")
    adv_h = adv_h.assign_coords(year=[year])
    
    all_years.append(adv_h)
    
    print('Finished computing horizontal advection of ', year)

# concat on a year dimension
adv_h_all = xr.concat(all_years, dim="year")
print(adv_h_all)


#%%
import pandas as pd 

adv_h_all = adv_h_all.stack(time_new=("year", "time"))
dates = pd.to_datetime(
    [f"{year}-{int(month):02d}-01" for year, month in zip(adv_h_all['year'].values, adv_h_all['time'].values)],
    format='%Y-%m-%d'
)
adv_h_all = adv_h_all.drop_vars(['time_new', 'year', 'time'])
adv_h_all.coords['time_new'] = ("time_new", dates)

#%%
adv_h_all.to_netcdf('/net/pyxis/data/varclim/stagiaires_alban/ENORA_JAMES/temperature_material_derivative_global/adv_h_100_200m_global_1993_2019.nc')

