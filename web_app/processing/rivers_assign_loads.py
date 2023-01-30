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

def process_loads_rec():
    conc0 = pd.read_csv(utils.conc_csv_path, usecols=['Indicator', 'nzsegment', 'lm1pred_01_2022']).dropna()

    conc0.rename(columns={'lm1pred_01_2022': 'init_conc', 'Indicator': 'indicator'}, inplace=True)

    conc1 = conc0.groupby(['indicator', 'nzsegment']).mean().reset_index()

    ## Clean up data - Excessive max values
    conc1.loc[(conc1.indicator == 'EC') & (conc1.init_conc > 1000)] = np.nan
    conc1.loc[(conc1.indicator == 'NO') & (conc1.init_conc > 20)] = np.nan

    conc1 = conc1.dropna().copy()

    conc1['nzsegment'] = conc1['nzsegment'].astype('int32')

    ## Create rough values from all reaches per indicator
    grp1 = conc1.groupby('indicator')

    mean1 = grp1['init_conc'].mean().round(3)
    stdev1 = grp1['init_conc'].std()

    ## Assign init conc and errors to each catchment
    mapping = booklet.open(utils.river_reach_mapping_path)
    flows = booklet.open(utils.river_flows_rec_path)

    starts = list(mapping.keys())
    indicators = conc1.indicator.unique()

    load_dict = {ind: {} for ind in indicators}
    for ind, grp in grp1:
        miss_conc = mean1.loc[ind]
        for catch_id in starts:
            c_reaches = mapping[catch_id][catch_id]
            df1 = pd.DataFrame(c_reaches, columns=['nzsegment'])
            r_conc = pd.merge(df1, grp, on='nzsegment', how='left').set_index('nzsegment')['init_conc']

            null_rows = r_conc.isnull()
            r_conc.loc[null_rows] = miss_conc
            na_len = r_conc.isnull().sum()
            if na_len > 0:
                raise ValueError('What the heck!')

            # Calc the loads
            r_conc_dict = r_conc.to_dict()
            r_load_dict = {way_id: int(round(conc * flows[way_id])) for way_id, conc in r_conc_dict.items()}

            load_dict[ind].update(r_load_dict)

    load_list = []
    for ind, loads in load_dict.items():
        segs = np.array(list(loads.keys()), dtype='int32')
        values = np.array(list(loads.values()), dtype='int16')
        ds1 = xr.Dataset({'load': (('indicator', 'nzsegment'), np.expand_dims(values, axis=0))},
                         coords={'nzsegment': (('nzsegment'), segs),
                                 'indicator': (('indicator'), [ind])})
        load_list.append(ds1)

    h5 = hdf5tools.H5(load_list)

    h5.to_hdf5(utils.river_reach_loads_path)


def process_loads_area():
    conc0 = pd.read_csv(utils.conc_csv_path, usecols=['Indicator', 'nzsegment', 'lm1pred_01_2022']).dropna()

    conc0.rename(columns={'lm1pred_01_2022': 'init_conc', 'Indicator': 'indicator'}, inplace=True)

    conc1 = conc0.groupby(['indicator', 'nzsegment']).mean().reset_index()

    ## Clean up data - Excessive max values
    conc1.loc[(conc1.indicator == 'EC') & (conc1.init_conc > 1000)] = np.nan
    conc1.loc[(conc1.indicator == 'NO') & (conc1.init_conc > 20)] = np.nan

    conc1 = conc1.dropna().copy()

    conc1['nzsegment'] = conc1['nzsegment'].astype('int32')

    ## Create rough values from all reaches per indicator
    grp1 = conc1.groupby('indicator')

    mean1 = grp1['init_conc'].mean().round(3)
    stdev1 = grp1['init_conc'].std()

    ## Assign init conc and errors to each catchment
    mapping = booklet.open(utils.river_reach_mapping_path)
    flows = booklet.open(utils.river_flows_area_path)

    starts = list(mapping.keys())
    indicators = conc1.indicator.unique()

    load_dict = {ind: {} for ind in indicators}
    for ind, grp in grp1:
        miss_conc = mean1.loc[ind]
        for catch_id in starts:
            c_reaches = mapping[catch_id][catch_id]
            df1 = pd.DataFrame(c_reaches, columns=['nzsegment'])
            r_conc = pd.merge(df1, grp, on='nzsegment', how='left').set_index('nzsegment')['init_conc']

            null_rows = r_conc.isnull()
            r_conc.loc[null_rows] = miss_conc
            na_len = r_conc.isnull().sum()
            if na_len > 0:
                raise ValueError('What the heck!')

            # Calc the loads
            r_conc_dict = r_conc.to_dict()
            r_load_dict = {way_id: int(round(conc * flows[way_id])) for way_id, conc in r_conc_dict.items()}

            load_dict[ind].update(r_load_dict)

    load_list = []
    for ind, loads in load_dict.items():
        segs = np.array(list(loads.keys()), dtype='int32')
        values = np.array(list(loads.values()), dtype='int16')
        ds1 = xr.Dataset({'load': (('indicator', 'nzsegment'), np.expand_dims(values, axis=0))},
                         coords={'nzsegment': (('nzsegment'), segs),
                                 'indicator': (('indicator'), [ind])})
        load_list.append(ds1)

    h5 = hdf5tools.H5(load_list)

    h5.to_hdf5(utils.river_reach_loads_area_path)







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






























































































































