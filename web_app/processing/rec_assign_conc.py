#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:51:17 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
import utils

pd.options.display.max_columns = 10

#######################################
### Assign conc

def process_conc():
    list1 = utils.error_cats()

    conc0 = pd.read_csv(utils.conc_csv_path, usecols=['Indicator', 'nzsegment', 'lm1seRes', 'lm1pred_01_2022']).dropna()

    conc0.rename(columns={'lm1seRes': 'error', 'lm1pred_01_2022': 'init_conc', 'Indicator': 'indicator'}, inplace=True)

    conc1 = conc0.groupby(['indicator', 'nzsegment']).mean().reset_index()

    # reaches2 = utils.read_pkl_zstd(utils.output_path.joinpath(utils.rec_delin_file), True)

    mapping = utils.read_pkl_zstd(utils.output_path.joinpath(utils.reach_mapping_file), True)

    ## Clean up data - Excessive max values
    conc1.loc[(conc1.indicator == 'EC') & (conc1.init_conc > 1000)] = np.nan
    conc1.loc[(conc1.indicator == 'NO') & (conc1.init_conc > 15)] = np.nan

    conc1['error'] = conc1['error'].abs()

    conc1 = conc1.dropna()

    conc1['nzsegment'] = conc1['nzsegment'].astype('int32')

    ## Create rough values from all reaches per indicator
    grp1 = conc1.groupby('indicator')

    mean1 = grp1[['error', 'init_conc']].mean().round(3)
    stdev1 = grp1[['error', 'init_conc']].std()

    ## Assign init conc and errors to each catchment
    # conc1['error_cat'] = pd.cut(conc1['error'], list1, labels=list1[:-1])

    starts = list(mapping.keys())
    indicators = conc1.indicator.unique()

    conc_dict = {ind: {} for ind in indicators}
    for ind, grp in grp1:
        for catch_id in starts:
            c_reaches = mapping[catch_id][catch_id]
            c_conc = grp[grp.nzsegment.isin(c_reaches)]

            if c_conc.empty:
                mean2 = mean1.loc[ind]
            else:
                mean2 = c_conc[['error', 'init_conc']].mean().round(3)

            # conc_dict[ind][catch_id] = mean2.to_dict()
            conc_dict[ind][catch_id] = pd.cut(mean2[['error']], list1, labels=list1[:-1])[0]


    utils.write_pkl_zstd(conc_dict, utils.error_pkl_path)

    # conc_dict0 = utils.read_pkl_zstd(utils.conc_pkl_path, True)













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






























































































































