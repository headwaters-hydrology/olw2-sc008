#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:51:17 2022

@author: mike
"""
import os
import xarray as xr
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
import hdf5tools
import booklet

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

#######################################
### Assign conc


def eco_process_power_modelled():
    list1 = utils.error_cats(0.03, 5.6, 0.1)

    stdev0 = pd.read_csv(utils.eco_catch_stdev_path)

    ## Create rough values from all reaches per indicator
    grp1 = stdev0.groupby('indicator')

    median1 = grp1['stdev'].median().round(3)

    ## Assign init conc and errors to each catchment
    mapping = booklet.open(utils.river_reach_mapping_path)

    starts = list(mapping.keys())
    indicators = stdev0.indicator.unique()

    # error_set = set()
    error_dict = {ind: {} for ind in indicators}
    for ind, grp in grp1:
        miss_error = median1.loc[ind]
        for catch_id in starts:
            c_reaches = mapping[catch_id][catch_id]
            df1 = pd.DataFrame(c_reaches, columns=['nzsegment'])
            r_errors = pd.merge(df1, grp, on='nzsegment').set_index('nzsegment')['stdev']
            if not r_errors.empty:
                null_rows = r_errors.isnull()
                r_errors.loc[null_rows] = miss_error
                r_errors[r_errors <= list1[0]] = list1[0] *1.1
                r_errors[r_errors > list1[-1]] = list1[-1]
                # error_set.update(set((r_errors * 1000).round().tolist()))

                r_errors1 = pd.cut(r_errors, list1, labels=list1[:-1])
                na_len = r_errors1.isnull().sum()
                if na_len > 0:
                    raise ValueError('What the heck!')
                error_dict[ind].update(r_errors1.to_dict())

    river_sims = xr.open_dataset(utils.eco_sims_catch_h5_path, engine='h5netcdf')
    river_sims['n_samples'] = river_sims.n_samples.astype('int16')
    river_sims.n_samples.encoding = {}
    river_sims['conc_perc'] = river_sims.conc_perc.astype('int8')
    river_sims.conc_perc.encoding = {}
    river_sims['error'] = river_sims.error.astype('int16')
    river_sims.error.encoding = {}
    river_sims['power'] = river_sims.power.astype('int8')
    river_sims['power'].encoding = {}

    error_list = []
    for ind in error_dict:
        errors0 = error_dict[ind]
        segs = np.array(list(errors0.keys()), dtype='int32')
        values = (np.array(list(errors0.values())) * 1000).astype(int)

        river_sims1 = river_sims.sel(error=values).copy()
        river_sims1 = river_sims1.rename({'error': 'nzsegment'})
        river_sims1['error'] = river_sims1['nzsegment']
        river_sims1['nzsegment'] = segs
        river_sims1 = river_sims1.assign_coords(indicator=ind).expand_dims('indicator')
        river_sims1 = river_sims1.drop('error')
        river_sims1['power'] = river_sims1['power'].astype(float)

        error_list.append(river_sims1)

    combo = utils.xr_concat(error_list)
    combo['power'].encoding = {'scale_factor': 1, '_FillValue': -99, 'dtype': 'int8'}

    hdf5tools.xr_to_hdf5(combo, utils.eco_power_model_path)






















































































































