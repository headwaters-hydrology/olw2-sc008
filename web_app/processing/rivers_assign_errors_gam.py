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

def process_errors():
    list1 = utils.log_error_cats(0.01, 3.43, 0.1)

    conc0 = pd.read_csv(utils.conc_csv_path, usecols=['Indicator', 'nzsegment', 'gam1seRes']).dropna()

    conc0.rename(columns={'gam1seRes': 'error', 'Indicator': 'indicator'}, inplace=True)

    conc1 = conc0.groupby(['indicator', 'nzsegment']).mean().reset_index()
    # conc1.loc[(conc1.indicator == 'EC') & (conc1.init_conc > 1000)] = np.nan
    # conc1.loc[(conc1.indicator == 'NO') & (conc1.init_conc > 20)] = np.nan

    conc1['error'] = conc1['error'].abs()

    conc1 = conc1.dropna()

    conc1['nzsegment'] = conc1['nzsegment'].astype('int32')

    ## Create rough values from all reaches per indicator
    grp1 = conc1.groupby('indicator')

    median1 = grp1['error'].median().round(3)

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
            r_errors[r_errors <= list1[0]] = list1[0] * 1.1
            r_errors[r_errors > list1[-1]] = list1[-1]
            # error_set.update(set((r_errors * 1000).round().tolist()))

            r_errors1 = pd.cut(r_errors, list1, labels=list1[:-1])
            na_len = r_errors1.isnull().sum()
            if na_len > 0:
                raise ValueError('What the heck!')
            error_dict[ind].update(r_errors1.to_dict())

    river_sims = xr.open_dataset(utils.river_sims_gam_path, engine='h5netcdf')
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
        river_sims1['error'] = segs
        river_sims1 = river_sims1.rename({'error': 'nzsegment'})
        river_sims1 = river_sims1.assign_coords(indicator=ind).expand_dims('indicator')

        error_list.append(river_sims1)

    combo = utils.xr_concat(error_list)

    hdf5tools.xr_to_hdf5(combo, utils.river_reach_error_gam_path)











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



























































































































