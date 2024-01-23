#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:43:40 2022

@author: mike
"""
import pandas as pd
import os
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

n_samples_year = [1, 4, 12, 26, 52]
n_years = utils.n_years
n_sims = 10000

list1 = utils.error_cats(0.01, 15.31, 0.2)
list1.insert(0, 0.001)

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for error in list1[:-1]:
            f = executor.submit(utils.power_sims_gw, error, n_years, n_samples_year, n_sims, utils.gw_sims_path)
            futures.append(f)
        runs = concurrent.futures.wait(futures)

    paths = [r.result() for r in runs[0]]
    paths.sort()

    h5 = hdf5tools.H5(paths)
    h5.to_hdf5(utils.gw_sims_h5_path)

    ## Remove temp files
    for path in paths:
        os.remove(path)
