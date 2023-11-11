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

base_path = pathlib.Path('/home/mike/data/OLW')

lakes_sims_h5_path = '/home/mike/data/OLW/web_app/output/lakes_sims/lakes_sims.h5'
lakes_stdev_moni_path = '/home/mike/data/OLW/web_app/output/lakes_stdev_monitored_v06.csv'

output_csv = 'power_calcs_at_monitored_lakes_v02.csv'

conc_perc = [70]

n_years = [5, 20]
n_samples_year = [6, 12, 26, 52, 104, 364]

n_samples0 = hdf5tools.utils.cartesian([n_years, n_samples_year])
n_samples = np.prod(n_samples0, axis=1)

n_samples_df = pd.DataFrame(n_samples0, columns=['n_years', 'n_samples_year'])
n_samples_df['n_samples'] = n_samples

start = 0.1
end = 1.6
step = 0.04


def log_error_cats(start, end, change):
    """

    """
    s1 = np.asarray(start).round(3)
    list1 = [s1]

    while s1 < end:
        delta = change
        s1 = round(s1 + delta, 3)
        list1.append(s1)

    return list1


#####################################################
### Process data

list1 = log_error_cats(start, end, step)

lake_sims = xr.open_dataset(lakes_sims_h5_path, engine='h5netcdf')

lake_sims1 = lake_sims.sel(conc_perc=conc_perc, n_samples=n_samples)

lake_stdev0 = pd.read_csv(lakes_stdev_moni_path)

nan_errors = lake_stdev0[lake_stdev0.stdev.isnull()].LFENZID.values.astype(int)

lake_stdev0.loc[lake_stdev0.stdev <= list1[0], 'stdev'] = list1[0] *1.1
lake_stdev0.loc[lake_stdev0.stdev > list1[-1], 'stdev'] = list1[-1]
lake_stdev0.loc[lake_stdev0.stdev.isnull(), 'stdev'] = list1[-1]

errors1 = (pd.cut(lake_stdev0.stdev, list1, labels=list1[:-1]).to_numpy() * 1000).astype(int)
lake_stdev0['error'] = errors1

lake_sims2 = lake_sims1.to_dataframe().reset_index()

power_data0 = pd.merge(lake_stdev0, lake_sims2, on='error').drop('error', axis=1)
power_data0['conc_perc'] = 100 - power_data0['conc_perc']
power_data0 = power_data0.rename(columns={'conc_perc': 'reduction'})
power_data0['stdev'] = power_data0['stdev'].round(3)

power_data1 = pd.merge(n_samples_df, power_data0, on='n_samples').drop('n_samples', axis=1)

power_data1.set_index(['indicator', 'lawa_id', 'site_id', 'LFENZID', 'reduction', 'n_years', 'n_samples_year']).to_csv(base_path.joinpath(output_csv))


