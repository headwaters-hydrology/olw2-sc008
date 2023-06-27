#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:43:40 2022

@author: mike
"""
import io
import zstandard as zstd
import codecs
import pickle
import pandas as pd
import numpy as np
# import requests
import xarray as xr
import orjson
# from shapely.geometry import shape, mapping
# import tethysts
import os
import geopandas as gpd
from scipy import stats
import base64
import pathlib
import h5py
import hdf5tools
import concurrent.futures
import multiprocessing as mp
import booklet

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

######################################################
### Sims

# catch_id = 3076139

reduction_ratios = range(10, 101, 10)
feature = 'rivers'

with booklet.open(utils.river_reach_mapping_path) as f:
    catches = list(f.keys())

if __name__ == '__main__':

    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for catch_id in catches[:160]:
            f = executor.submit(utils.calc_river_reach_reductions, feature, catch_id, reduction_ratios)
            futures.append(f)
        runs = concurrent.futures.wait(futures)

    files = [r.result() for r in runs[0]]

    combo1 = utils.xr_concat(files)
    hdf5tools.xr_to_hdf5(combo1, utils.river_reductions_model_path)

    # h5 = hdf5tools.H5(files)
    # h5.to_hdf5(utils.river_reductions_model_path)

