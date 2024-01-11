#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:40:37 2023

@author: mike
"""
import os, pathlib
import pandas as pd
import numpy as np
import geopandas as gpd

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

####################################################
### Parameters

lake_ecoli_limits = {
    'A': {
        'E. coli G540': (-1, 0.05),
        'E. coli G260': (-1, 0.2),
        'E. coli median': (-1, 130),
        'E. coli Q95': (-1, 540)
        },
    'B': {
        'E. coli G540': (-1, 0.1),
        'E. coli G260': (-1, 0.3),
        'E. coli median': (-1, 130),
        'E. coli Q95': (-1, 1000)
        },
    'C': {
        'E. coli G540': (-1, 0.2),
        'E. coli G260': (-1, 0.34),
        'E. coli median': (-1, 130),
        'E. coli Q95': (-1, 1200)
        },
    'D': {
        'E. coli G540': (-1, 0.3),
        'E. coli G260': (-1, 1.01),
        'E. coli median': (-1, 1000000),
        'E. coli Q95': (-1, 1000000)
        },
    'E': {
        'E. coli G540': (-1, 1.01),
        'E. coli G260': (-1, 1.01),
        'E. coli median': (-1, 1000000),
        'E. coli Q95': (-1, 1000000)
        }
    }

lake_ammonia_limits = {
    'A': {'Ammoniacal nitrogen median': (-1, 0.03)},
    'B': {'Ammoniacal nitrogen median': (-1, 0.24)},
    'C': {'Ammoniacal nitrogen median': (-1, 1.3)},
    'D': {'Ammoniacal nitrogen median': (-1, 100000)}
    }

lake_tp_limits = {
    'A': {'Total phosphorus median': (-1, 10)},
    'B': {'Total phosphorus median': (-1, 20)},
    'C': {'Total phosphorus median': (-1, 50)},
    'D': {'Total phosphorus median': (-1, 100000)}
    }

lake_chla_limits = {
    'A': {'Chla median': (-1, 2),
          'Chla max': (-1, 10)},
    'B': {'Chla median': (-1, 5),
          'Chla max': (-1, 25)},
    'C': {'Chla median': (-1, 12),
          'Chla max': (-1, 60)},
    'D': {'Chla median': (-1, 100000),
          'Chla max': (-1, 100000)}
    }

lake_tn_limits = {
    'A': {
        True: (-1, 160),
        False: (-1, 300),
        },
    'B': {
        True: (-1, 350),
        False: (-1, 500),
        },
    'C': {
        True: (-1, 750),
        False: (-1, 800),
        },
    'D': {
        True: (-1, 100000),
        False: (-1, 100000),
        }
    }

lake_cyano_limits = {
    'A': {'Total cyano Q80': (-1, 0.5),
          },
    'B': {'Total cyano Q80': (-1, 1),
          },
    'C': {'Total cyano Q80': (-1, 10),
          'Toxic cyano Q80': (-1, 1.8)},
    'D': {'Total cyano Q80': (-1, 100000),
          'Toxic cyano Q80': (-1, 100000)}
    }


bottom_line_limits = {
    ('lake', 'Ammonia'): {'Ammoniacal nitrogen median': (-1, 0.24),
                          # 'Ammoniacal nitrogen Q95': (-1, 0.40)
                          },
    # ('lake', 'Cyano'): {
    #     'Total cyano Q80': (-1, 10),
    #     'Toxic cyano Q80': (-1, 1.8)
    #     },
    ('lake', 'Chla'): {
        'Chla median': (-1, 12),
        # 'Chla max': (-1, 60)
        },
    # ('lake', 'Chla'): {
    #     'Chla median': (-1, 12),
    #     },
    ('lake', 'Total nitrogen'): {
        True: (-1, 750),
        False: (-1, 800),
        },
    ('lake', 'Total phosphorus'): {'Total phosphorus median': (-1, 50)},
    }


parameter_special_cols_dict = {
    ('lake', 'Total nitrogen'): ['stratified', 'Total nitrogen median'],
    }

parameter_limits_dict = {
    ('lake', 'Ammonia'): lake_ammonia_limits,
    # ('lake', 'Cyano'): lake_cyano_limits,
    ('lake', 'Chla'): lake_chla_limits,
    ('lake', 'Total nitrogen'): lake_tn_limits,
    ('lake', 'Total phosphorus'): lake_tp_limits,
    # ('lake', 'E.coli'): lake_ecoli_limits,
    }

param_mapping = {'CHLA': 'Chla', 'NH4N': 'Ammonia', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus', 'ECOLI': 'E.coli'}

####################################################
### Functions


def get_required_cols(limits):
    """

    """
    cols = set()
    for band, limit in limits.items():
        cols.update(set(list(limit.keys())))

    return list(cols)


def assign_bands_normal(feature, parameter, data, limits):
    """

    """
    b0 = data[['nzsegment']].copy()
    b0[parameter] = 'NA'
    bool_series = b0.set_index(['nzsegment'])[parameter]

    for band, limit in reversed(limits.items()):
        bool_list = []
        for col, minmax in limit.items():
            min1, max1 = minmax
            bool0 = (data[col] > min1) & (data[col] <= max1)
            bool_list.append(bool0)

        bool1 = pd.concat(bool_list, axis=1)
        bool_series[bool1.all(axis=1).values] = band

    return bool_series


def assign_bands_for_parameter(feature, parameter, data):
    """

    """
    key = (feature, parameter)

    ## Get the limits
    limits = parameter_limits_dict[key]

    ## Check that the data has the required cols
    if parameter in parameter_special_cols_dict:
        cols = parameter_special_cols_dict[parameter].copy()
    else:
        cols = get_required_cols(limits)

    if not all(np.in1d(cols, data.columns)):
        raise ValueError('Not all required columns are in the data.')

    ## Run the calcs
    if key in parameter_special_cols_dict:
        b0 = data[cols[:2]].copy()
        b0[parameter] = 'NA'
        bool_series = b0.set_index(cols[:2])[parameter].sort_index()
        for band, limit in reversed(limits.items()):
            bool_list = []
            for col, minmax in limit.items():
                min1, max1 = minmax
                bool0 = (data[cols[-1]] >= min1) & (data[cols[-1]] < max1)
                bool0.name = col
                bool_list.append(bool0)

            bool1 = pd.concat(bool_list, axis=1)
            bool1['nzsegment'] = data['nzsegment']
            bool2 = bool1.set_index('nzsegment').stack().sort_index()
            bool2.index.names = cols[:2]

            bool_series.loc[bool_series.index.isin(bool2.loc[bool2].index)] = band

        data0 = pd.merge(data[cols[:2]], bool_series.reset_index(), on=cols[:2], how='left').drop(cols[1], axis=1).set_index('nzsegment')
    else:
        data0 = assign_bands_normal(parameter, data, limits)

    return data0



####################################################
### Process data

lakes0 = gpd.read_file(utils.lakes_fenz_poly_path)
lakes1 = lakes0[['LID', 'MaxDepth']].rename(columns={'MaxDepth': 'max_depth', 'LID': 'LFENZID'}).dropna().copy()
lakes1['LFENZID'] = lakes1['LFENZID'].astype('int32')
lakes1['max_depth'] = lakes1['max_depth'].round().astype('int32')

lakes1['stratified'] = lakes1['max_depth'] > 20

# .to_feather(b1, compression='zstd', compression_level=1)

data1 = pd.read_csv(utils.lakes_conc_moni_path).rename(columns={'indicator': 'parameter', 'conc_2020': 'conc'})

# data0['current_conc'] = np.exp(data0['current_conc'])
# data1 = data0[data0.indicator.isin(list(param_mapping.keys()))].rename(columns={'indicator': 'parameter'}).replace({'parameter': param_mapping}).copy()

data2 = pd.merge(data1, lakes1, on='LFENZID')

results_list = []
for key, limit in bottom_line_limits.items():
    feature = key[0]
    param = key[1]

    # d1 = data2[data2.parameter.str.contains(param)].set_index(['lawa_id', 'parameter'])

    if key in parameter_special_cols_dict:
        param2 = parameter_special_cols_dict[key][1]
        cat_col = parameter_special_cols_dict[key][0]

        d1 = data2[data2.parameter == param2].copy()

        d1['limit'] = limit[False][1]
        d1.loc[d1[cat_col], 'limit'] = limit[True][1]

        d1['perc_reduction'] = 1 - (d1['limit']/d1['conc'])
        results_list.append(d1)
    else:
        for param2 in limit:
            d1 = data2[data2.parameter == param2].copy()
            d1['limit'] = limit[param2][1]

            d1['perc_reduction'] = 1 - (d1['limit']/d1['conc'])
            results_list.append(d1)

results1 = pd.concat(results_list)
results1.loc[results1['perc_reduction'] < 0, 'perc_reduction'] = 0
results1['perc_reduction'] = results1['perc_reduction'].round(3)
results1['conc'] = results1['conc'].round(4)

results2 = results1.drop(['max_depth', 'stratified'], axis=1)

results2.to_csv('/home/mike/data/OLW/web_app/lakes/lakes_monitored_perc_reductions_required.csv', index=False)


















