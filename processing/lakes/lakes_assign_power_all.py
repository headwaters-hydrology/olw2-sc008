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
import multiprocessing as mp
import concurrent.futures
import geobuf

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing

encodings = {'power': {'scale_factor': 1, '_FillValue': -99, 'dtype': 'int8'},
             }

# start = 0.02
# end = 1.4
# step = 0.03

start = 0.1
end = 1.7
step = 0.04


def lakes_process_power_monitored():
    # start = 0.069
    # end = 4.299

    list1 = utils.log_error_cats(start, end, step)

    # lakes0 = xr.open_dataset(utils.lakes_stdev_moni_path, engine='h5netcdf')
    lakes1 = pd.read_csv(utils.lakes_stdev_moni_path)
    # lakes0b = lakes0.groupby(['indicator', 'site_id'])['stdev'].mean().reset_index()

    ## Combine with sims
    lake_sims = xr.load_dataset(utils.lakes_sims_h5_path, engine='h5netcdf')

    grp = lakes1.groupby('indicator')

    error_list = []
    for ind, data in grp:
        errors = data.drop('indicator', axis=1)
        # names = data.LFENZID.values.astype(int)
        # errors = data.stdev.copy()
        nan_errors = errors[errors.stdev.isnull()].site_id.values

        errors.loc[errors.stdev <= list1[0], 'stdev'] = list1[0] *1.1
        errors.loc[errors.stdev > list1[-1], 'stdev'] = list1[-1]
        errors.loc[errors.stdev.isnull(), 'stdev'] = list1[-1]

        errors1 = (pd.cut(errors.stdev, list1, labels=list1[:-1]).to_numpy() * 1000).astype(int)

        lake_sims1 = lake_sims.sel(error=errors1).copy()
        lake_sims1['error'] = errors.site_id.values
        lake_sims1 = lake_sims1.rename({'error': 'site_id', 'power': 'power_monitored'})
        lake_sims2 = xr.where(lake_sims1.site_id.isin(nan_errors), np.nan, lake_sims1)
        lake_sims2['conc_perc'] = lake_sims2['conc_perc'].astype('int8')
        lake_sims2['n_samples'] = lake_sims2['n_samples'].astype('int16')
        lake_sims2 = lake_sims2.assign(LFENZID=(('site_id'), errors.LFENZID.values))
        lake_sims2['LFENZID'] = lake_sims2['LFENZID'].astype('int32')
        # lake_sims2 = lake_sims2.assign(lawa_id=(('site_id'), errors.lawa_id.values))
        lake_sims2['power_monitored'].encoding = encodings['power']

        lake_sims2 = lake_sims2.assign_coords(indicator=ind).expand_dims('indicator')

        error_list.append(lake_sims2)

    h5 = hdf5tools.H5(error_list)

    h5.to_hdf5(utils.lakes_power_moni_path)


def lakes_process_power_modelled():
    # start = 0.069
    # end = 4.299

    list1 = utils.log_error_cats(start, end, step)

    lakes0 = xr.open_dataset(utils.lakes_stdev_model_path, engine='h5netcdf')
    lakes1 = lakes0.sel(model='BoostingRegressor', drop=True)
    # lakes1 = lakes0.sel(model='RandomForestRegressor', drop=True)

    ## Combine with sims
    lake_sims = xr.load_dataset(utils.lakes_sims_h5_path, engine='h5netcdf')

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
        lake_sims1 = lake_sims1.rename({'error': 'LFENZID', 'power': 'power_modelled'})
        lake_sims1['conc_perc'] = lake_sims1['conc_perc'].astype('int8')
        lake_sims1['n_samples'] = lake_sims1['n_samples'].astype('int16')
        lake_sims1['LFENZID'] = lake_sims1['LFENZID'].astype('int32')
        lake_sims1['power_modelled'].encoding = encodings['power']

        lake_sims1 = lake_sims1.assign_coords(indicator=ind).expand_dims('indicator')

        error_list.append(lake_sims1)

    h5 = hdf5tools.H5(error_list)

    h5.to_hdf5(utils.lakes_power_model_path)



















