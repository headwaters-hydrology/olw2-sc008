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

import utils

pd.options.display.max_columns = 10

#######################################
### Assign conc

def gw_process_errors_points():
    list1 = utils.error_cats(0.01, 15.31, 0.1)
    list1.insert(0, 0.001)

    # errors0 = xr.open_dataset(utils.gw_monitoring_data_path)
    # for param in errors0:
    #     attrs = errors0[param].attrs
    #     val = errors0[param]
    #     if 'scale' in attrs:
    #         val = val * attrs['scale']
    #     if 'offset' in attrs:
    #         val = val + attrs['offset']
    #     errors0[param] = val
    #     errors0[param].attrs = attrs

    errors0 = pd.read_hdf(utils.gw_data_path, key='data')[['use_std']].rename(columns={'use_std': 'error'})
    errors0['indicator'] = 'Nitrate'

    errors1 = errors0.copy()

    ## Create rough values from all reaches per indicator
    grp1 = errors1.groupby('indicator')

    median1 = grp1['error'].median().round(3)

    ## Assign init conc and errors to each location
    # indicators = errors1.indicator.unique()

    error_dict = {}
    for ind, errors in grp1:
        miss_error = median1.loc[ind]

        null_rows = errors.error.isnull()
        errors.loc[null_rows, 'error'] = miss_error

        errors.loc[errors.error <= list1[0], 'error'] = list1[0]*1.05
        errors.loc[errors.error > list1[-1], 'error'] = list1[-1]

        errors2 = pd.cut(errors.error, list1, labels=list1[:-1])

        error_dict[ind] = errors2

    gw_sims = xr.open_dataset(utils.gw_sims_h5_path, engine='h5netcdf')
    gw_sims['n_samples'] = gw_sims.n_samples.astype('int16')
    gw_sims.n_samples.encoding = {}
    gw_sims['conc_perc'] = gw_sims.conc_perc.astype('int8')
    gw_sims.conc_perc.encoding = {}
    gw_sims['error'] = gw_sims.error.astype('int16')
    gw_sims.error.encoding = {}
    gw_sims['power'] = gw_sims.power.astype('int8')
    gw_sims['power'].encoding = {}

    error_list = []
    for ind in error_dict:
        errors0 = error_dict[ind]
        refs = np.array(errors0.index)
        values = (np.asarray(errors0) * 1000).astype(int)

        gw_sims1 = gw_sims.sel(error=values).copy()
        gw_sims1 = gw_sims1.rename({'error': 'ref'})
        gw_sims1['error'] = gw_sims1['ref']
        gw_sims1['ref'] = refs
        gw_sims1 = gw_sims1.assign_coords(indicator=ind).expand_dims('indicator')
        gw_sims1 = gw_sims1.drop('error')

        error_list.append(gw_sims1)

    combo = utils.xr_concat(error_list)

    hdf5tools.xr_to_hdf5(combo, utils.gw_points_error_path)











## Filling missing end segments - Too few data...this will need to be delayed...
# starts = reaches2.start.unique()

# grp1 = conc1.groupby('indicator')

# extra_rows = []
# problem_segs = []
# for ind, grp in grp1:
#     conc_segs = grp['nzsegment'].unique()
#     mis_segs = starts[~np.in1d(starts, conc_segs)]
#     for mis_seg in mis_segs:
#         mis_map = mapping[mis_seg][mis_seg]

#         mis_conc = grp[grp.nzsegment.isin(mis_map[1:])]
#         mis_conc_segs = mis_conc.nzsegment.unique()

#         if len(mis_conc_segs) == 0:
#             # print(mis_seg)
#             # print('I have a problem...')
#             problem_segs.append(mis_seg)

#         for seg in mis_map[1:]:
#             if seg in mis_conc_segs:
#                 mis_conc1 = mis_conc[mis_conc.nzsegment == seg]
#                 extra_rows.append(mis_conc1)
#                 break


# def make_gis_file(output_path):
#     """

#     """
#     w0 = nzrec.Water(utils.nzrec_data_path)
#     node = w0._node
#     way = w0._way

#     data = []
#     geo = []
#     for i, row in conc1[conc1.indicator == 'BD'].iterrows():
#         nodes = way[row.nzsegment]
#         geo.append(LineString(np.array([node[int(i)] * 0.0000001 for i in nodes])))
#         data.append([row.nzsegment, row.error])

#     gdf = gpd.GeoDataFrame(data, columns = ['nzsegment', 'error'], geometry=geo, crs=4326)

#     gdf.to_file(output_path)



























































































































