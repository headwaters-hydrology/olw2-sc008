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
from gistools import vector
import geopandas as gpd
from scipy import stats
import base64
import pathlib
import h5py
import hdf5tools
import concurrent.futures
import multiprocessing as mp
import utils

pd.options.display.max_columns = 10

######################################################
### Sims

# catch_id = 14295077
n_samples_year = utils.n_samples_year
n_years = utils.n_years
n_sims = 10000
output_path = '/media/nvme1/data/OLW/web_app/output/sims'

# conc_dict0 = utils.read_pkl_zstd(utils.conc_pkl_path, True)

# concs_dict = {ind: v[catch_id] for ind, v in conc_dict0.items()}

# concs_dict = {}
# for ind, v in conc_dict0.items():
#     for catch_id, error in v.items():
#         if catch_id in concs_dict:
#             concs_dict[catch_id].update({ind: error})
#         else:
#             concs_dict[catch_id] = {ind: error}

# errors = set()
# for ind, v in conc_dict0.items():
#     for catch_id, error in v.items():
#         errors.add(int(error['error']*1000))


start = 0.025
list1 = [0.001, 0.005, 0.01, start]

while start < 2.7:
    if start < 0.1:
        start = round(start*2, 3)
    else:
        start = round(start*1.2, 3)
    list1.append(start)


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for error in list1[:-1]:
            f = executor.submit(utils.catch_sims, error, n_years, n_samples_year, n_sims, output_path)
            futures.append(f)
        runs = concurrent.futures.wait(futures)

    paths = [r.result() for r in runs[0]]
    paths.sort()

    h5 = hdf5tools.H5(paths)
    combo_path = os.path.join(output_path, 'sims.h5')
    h5.to_hdf5(combo_path)



















































































