#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:17:00 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import intersection
import hdf5tools
import xarray as xr
# import dbm
import shelflet
# import shelve
import multiprocessing as mp
import concurrent.futures
import geobuf

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing

lakes0 = pd.read_csv(utils.raw_lakes_path)

## Error assessments
lakes0['CV'] = lakes0.CV.round(3)

errors = lakes0['CV'].unique()
errors.sort()

n_samples_year = utils.n_samples_year
n_years = utils.n_years
n_sims = 10000


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for error in errors:
            f = executor.submit(utils.catch_sims, error, n_years, n_samples_year, n_sims, utils.lakes_sims_path)
            futures.append(f)
        runs = concurrent.futures.wait(futures)

    paths = [r.result() for r in runs[0]]
    paths.sort()

    h5 = hdf5tools.H5(paths)
    h5.to_hdf5(utils.lakes_sims_h5_path)




















































