#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:15:54 2023

@author: mike
"""
import sys
import os
import pandas as pd
import numpy as np
import pathlib
import geopandas as gpd
import booklet
import geobuf
from gistools import vector

if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

######################################################
### Parameters

#####################################################
### Process data


def eco_catchment_stdev_processing():
    """

    """
    data_list = []
    for param, val in utils.eco_catch_data_dict.items():
        sites0 = pd.read_csv(utils.eco_data_path.joinpath(val['file_name']))
        cols = [col for col in val.values() if col in sites0]
        sites1 = sites0[cols].dropna().rename(columns={v: k for k, v in val.items()}).copy()
        sites1 = sites1[(sites1.stdev > 0) & (sites1.nzsegment > 0)].copy()
        sites1['indicator'] = param
        data_list.append(sites1)

    stdev0 = pd.concat(data_list)
    stdev0['nzsegment'] = stdev0['nzsegment'].astype('int32')
    stdev0['stdev'] = stdev0['stdev']/stdev0['mean']
    stdev1 = stdev0.drop('mean', axis=1)

    with booklet.open(utils.river_catch_major_path) as f:
        segs = np.array(list(f.keys()))

    segs3 = np.repeat(segs, 3)
    ind3 = np.tile(np.array(list(utils.eco_catch_data_dict.keys())), len(segs))
    defaults = pd.DataFrame.from_dict(utils.eco_catch_stdev_defaults, orient='index', columns=['default'])
    defaults.index.name = 'indicator'
    defaults = defaults.reset_index()

    segs_df = pd.DataFrame(zip(segs3, ind3), columns=['nzsegment', 'indicator'])
    segs_df = pd.merge(segs_df, defaults, on='indicator')
    stdev2 = pd.merge(segs_df, stdev1, on=['nzsegment', 'indicator'], how='left')
    stdev2.loc[stdev2.stdev.isnull(), 'stdev'] = stdev2.loc[stdev2.stdev.isnull(), 'default']
    stdev3 = stdev2.drop('default', axis=1).copy()

    stdev3.to_csv(utils.eco_catch_stdev_path, index=False)





































































































