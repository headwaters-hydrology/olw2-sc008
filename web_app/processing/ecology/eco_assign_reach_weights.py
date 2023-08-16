#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:58:09 2023

@author: mike
"""
import booklet
import pandas as pd
import xarray as xr
import numpy as np
from hdf5tools import H5

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

#############################################
### Parameters

# catch_id = 3076139

# params = list(utils.indicator_dict.keys())

param_weights = {
    'peri': {
        'Total nitrogen': 0.4,
        'Total phosphorus': 0.4,
        'Visual Clarity': 0.2
        },
    'mci': {
        'Total nitrogen': 0.5,
        'Total phosphorus': 0.2,
        'Visual Clarity': 0.3
        },
    'sediment': {
        'Total nitrogen': 0,
        'Total phosphorus': 0,
        'Visual Clarity': 1
        },
    }

params = list(param_weights['peri'].keys())

#############################################
### Processing


def eco_calc_river_reach_weights():
    """

    """
    reach_red0 = xr.open_dataset(utils.river_reductions_model_path)

    reach_weights_list = []
    with booklet.open(utils.river_reach_mapping_path) as f:
        for catch_id, branches in f.items():
            branch = branches[catch_id]
            reach_red1 = reach_red0.sel(nzsegment=branch)[params].copy().load()

            arr_list = []
            for param, weights in param_weights.items():
                arr1 = np.ceil(sum([reach_red1[p]*w for p, w in weights.items()])).astype('int8')
                arr1.name = param
                arr_list.append(arr1)

            reach_weights0 = xr.merge(arr_list)
            reach_weights_list.append(reach_weights0)

    rw0 = H5(reach_weights_list)
    rw0.to_hdf5(utils.eco_reach_weights_h5_path)





































































