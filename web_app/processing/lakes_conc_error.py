#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:17:00 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import intersection
import hdf5tools
import xarray as xr
# import dbm
import shelflet
# import shelve
import multiprocessing as mp
import concurrent.futures
import geobuf

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing

lakes0 = pd.read_csv(utils.raw_lakes_path)
lakes_poly0 = gpd.read_file(utils.lakes_poly_path)
lakes_poly0['geometry'] = lakes_poly0.simplify(20)

lakes0 = lakes0.rename(columns={'SiteID': 'site_id'})

lakes1 = pd.merge(lakes_poly0.drop(['elevation', 'geometry'], axis=1), lakes0, on='site_id').drop('site_id', axis=1)
lakes1['CV'] = lakes1.CV.round(3)

## Combine with sims
lake_sims = xr.open_dataset(utils.lakes_sims_h5_path, engine='h5netcdf')

grp = lakes1.groupby('indicator')

error_list = []
for ind, data in grp:
    names = data.name.values
    values = (data.CV.values * 1000).astype(int)

    lake_sims1 = lake_sims.sel(error=values).copy()
    lake_sims1['error'] = names
    lake_sims1 = lake_sims1.rename({'error': 'name'})
    lake_sims1 = lake_sims1.assign_coords(indicator=ind).expand_dims('indicator')

    error_list.append(lake_sims1)

h5 = hdf5tools.H5(error_list)

h5.to_hdf5(utils.lakes_error_path)





















































