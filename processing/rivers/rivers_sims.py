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

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

######################################################
### Sims

# catch_id = 14295077
n_samples_year = utils.n_samples_year
n_years = utils.n_years
n_sims = 10000

list1 = utils.log_error_cats(0.14, 1.55, 0.03)


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for error in list1:
            f = executor.submit(utils.power_sims_rivers, error, n_years, n_samples_year, n_sims, utils.river_sims_path)
            futures.append(f)
        runs = concurrent.futures.wait(futures)

    paths = [r.result() for r in runs[0]]
    paths.sort()

    h5 = hdf5tools.H5(paths)
    h5.to_hdf5(utils.river_sims_h5_path)

    ## Remove temp files
    for path in paths:
        os.remove(path)
