#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 07:39:50 2023

@author: mike
"""
import os
import numpy as np
import pandas as pd
import pathlib
import xarray as xr
import hdf5tools
import hdf5plugin
import nzrec

pd.options.display.max_columns = 10

######################################################
### Parameters

base_path = pathlib.Path('/home/mike/git/olw-repos/olw2-sc008/web_app/app/assets')

lakes_power_h5_path = base_path.joinpath('lakes_power_monitored.h5')

output_csv = base_path.joinpath('power_calcs_at_monitored_lakes.csv.zip')

# conc_perc = [70]

# n_years = [5, 20]
# n_samples_year = [6, 12, 26, 52, 104, 364]

# n_samples0 = hdf5tools.utils.cartesian([n_years, n_samples_year])
# n_samples = np.prod(n_samples0, axis=1)

# n_samples_df = pd.DataFrame(n_samples0, columns=['n_years', 'n_samples_year'])
# n_samples_df['n_samples'] = n_samples

# start = 0.1
# end = 1.6
# step = 0.04


# def log_error_cats(start, end, change):
#     """

#     """
#     s1 = np.asarray(start).round(3)
#     list1 = [s1]

#     while s1 < end:
#         delta = change
#         s1 = round(s1 + delta, 3)
#         list1.append(s1)

#     return list1


#####################################################
### Process data

# list1 = log_error_cats(start, end, step)

power1 = xr.open_dataset(lakes_power_h5_path, engine='h5netcdf')

power2 = power1.to_dataframe().dropna().reset_index()

power2.to_csv(output_csv, index=False)



























































