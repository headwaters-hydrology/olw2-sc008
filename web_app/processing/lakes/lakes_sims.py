#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:17:00 2022

@author: mike
"""
import os
import pathlib
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import intersection
import hdf5tools
import xarray as xr
# import dbm
# import shelve
import multiprocessing as mp
import concurrent.futures
import geobuf
import sys

if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing

## Error assessments
# indicators = ['ECOLI', 'Secchi', 'TN', 'TP', 'CHLA', 'NH4N', 'CYANOTOT']

# lakes0 = xr.open_dataset(utils.lakes_stdev_path, engine='h5netcdf')

# lakes1 = lakes0.sel(model='BoostingRegressor', indicator=indicators)

# start = lakes1.stdev.min().round(3).values
# end = lakes1.stdev.max().round(3).values

# start = 0.069
# end = 4.299
start = 0.1
end = 1.7

errors = utils.log_error_cats(start, end, 0.04)

# n_samples_year = utils.n_samples_year
n_samples_year = [4, 6, 12, 26, 52, 104, 364]
n_years = utils.n_years
n_sims = 10000


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for error in errors[:-1]:
            f = executor.submit(utils.power_sims_lakes, error, n_years, n_samples_year, n_sims, utils.lakes_sims_path)
            futures.append(f)
        runs = concurrent.futures.wait(futures)

    paths = [r.result() for r in runs[0]]
    paths.sort()

    h5 = hdf5tools.H5(paths)
    h5.to_hdf5(utils.lakes_sims_h5_path)

    ## Remove temp files
    for path in paths:
        os.remove(path)



# base_path = pathlib.Path('/home/mike/data/OLW/web_app/output/lakes_sims')

# paths = []
# for path in base_path.iterdir():
#     try:
#         _ = int(path.stem)
#         paths.append(path)
#     except:
#         pass

# paths.sort()












































