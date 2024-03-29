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

# indicators = ['BD', 'DR', 'EC', 'NH', 'NO', 'TN', 'TP', 'TU']

error_name = '_new_gam_seRes'


def rivers_process_power_modelled():
    # list1 = utils.log_error_cats(0.01, 2.72, 0.1)
    # list1 = utils.log_error_cats(0.01, 3.43, 0.1)
    # list1 = utils.log_error_cats(0.01, 2.55, 0.05)
    # list1 = [0.001] + list1
    list1 = utils.log_error_cats(0.14, 1.55, 0.03)

    # conc0 = pd.read_csv(utils.conc_csv_path, usecols=['Indicator', 'nzsegment', 'lm1seRes']).dropna()
    # conc0.rename(columns={'lm1seRes': 'error', 'Indicator': 'indicator'}, inplace=True)

    conc0 = pd.read_csv(utils.river_errors_model_path)
    conc0a = conc0.set_index('nzsegment').loc[:, [col for col in conc0.columns if error_name in col]].stack()
    conc0a.name = 'error'
    conc0b = conc0a.reset_index()
    conc0b['indicator'] = conc0b['level_1'].apply(lambda x: x.split(error_name)[0])
    conc0b = conc0b.drop('level_1', axis=1)

    conc1 = conc0b.groupby(['indicator', 'nzsegment']).mean().reset_index()
    # conc1.loc[(conc1.indicator == 'EC') & (conc1.init_conc > 1000)] = np.nan
    # conc1.loc[(conc1.indicator == 'NO') & (conc1.init_conc > 20)] = np.nan

    conc1['error'] = conc1['error'].abs()

    conc1 = conc1.dropna()

    conc1['nzsegment'] = conc1['nzsegment'].astype('int32')

    ## Create rough values from all reaches per indicator
    grp1 = conc1.groupby('indicator')

    median1 = grp1['error'].median().round(2)

    ## Assign init conc and errors to each catchment
    mapping = booklet.open(utils.river_reach_mapping_path)

    starts = list(mapping.keys())
    indicators = conc1.indicator.unique()

    # error_set = set()
    error_dict = {ind: {} for ind in indicators}
    for ind, grp in grp1:
        miss_error = median1.loc[ind]
        for catch_id in starts:
            c_reaches = mapping[catch_id][catch_id]
            df1 = pd.DataFrame(c_reaches, columns=['nzsegment'])
            r_errors = pd.merge(df1, grp, on='nzsegment', how='left').set_index('nzsegment')['error']
            # c_conc = grp[grp.nzsegment.isin(c_reaches)]

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

    river_sims = xr.open_dataset(utils.river_sims_h5_path, engine='h5netcdf')
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

        error_list.append(river_sims1)

    combo = utils.xr_concat(error_list)
    combo['power'].encoding = {'scale_factor': 1, '_FillValue': -99, 'dtype': 'int8'}

    hdf5tools.xr_to_hdf5(combo, utils.river_power_model_path)






















































































































