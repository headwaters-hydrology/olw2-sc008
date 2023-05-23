#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:17:00 2022

@author: mike
"""
import sys
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

if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing


def lakes_conc_error_processing():
    start = 0.069
    end = 4.299

    list1 = utils.log_error_cats(start, end, 0.1)

    lakes0 = xr.open_dataset(utils.lakes_stdev_all_path, engine='h5netcdf')
    lakes1 = lakes0.sel(model='BoostingRegressor', drop=True)

    lakes_poly = gpd.read_feather(utils.lakes_poly_path)

    lakes1 = lakes1.where(lakes1.LFENZID.isin(lakes_poly.LFENZID.values), drop=True)

    ## Combine with sims
    lake_sims = xr.open_dataset(utils.lakes_sims_h5_path, engine='h5netcdf')

    grp = lakes1.groupby('indicator', squeeze=True)

    error_list = []
    for ind, data in grp:
        data = data.squeeze(drop=True)
        names = data.LFENZID.values.astype(int)
        errors = data.stdev

        errors[errors <= list1[0]] = list1[0] *1.1
        errors[errors > list1[-1]] = list1[-1]
        # error_set.update(set((r_errors * 1000).round().tolist()))

        errors1 = (pd.cut(errors, list1, labels=list1[:-1]).to_numpy() * 1000).astype(int)

        lake_sims1 = lake_sims.sel(error=errors1).copy()
        lake_sims1['error'] = names
        lake_sims1 = lake_sims1.rename({'error': 'LFENZID'})
        lake_sims1 = lake_sims1.assign_coords(indicator=ind).expand_dims('indicator')

        lake_sims1['LFENZID'] = lake_sims1['LFENZID'].astype('int32')

        error_list.append(lake_sims1)

    h5 = hdf5tools.H5(error_list)

    h5.to_hdf5(utils.lakes_power_all_path)





















































