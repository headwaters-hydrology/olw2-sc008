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
lakes_nps_path = '/home/mike/data/OLW/web_app/lakes/lakes_monitored_perc_reductions_required.csv'
lakes_stdev_moni_path = '/home/mike/data/OLW/web_app/output/lakes_stdev_monitored_v06.csv'

output_csv = 'power_calcs_at_monitored_lakes_for_bottom_line_v02.csv'
output2_csv = 'n_samples_at_monitored_lakes_for_bottom_line_gte_80_power_v02.csv'

n_years = [5, 20]
n_samples_year = [6]

n_samples0 = hdf5tools.utils.cartesian([n_years, n_samples_year])
n_samples = np.prod(n_samples0, axis=1)

n_samples_df = pd.DataFrame(n_samples0, columns=['n_years', 'n_samples_year'])
n_samples_df['n_samples'] = n_samples

start = 0.1
end = 1.7
step = 0.04

param_mapping = {'CHLA': 'Chla median', 'NH4N': 'Ammoniacal nitrogen median', 'TN': 'Total nitrogen median', 'TP': 'Total phosphorus median'}


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
### Estimate power

list1 = log_error_cats(start, end, step)

lake_stdev0 = pd.read_csv(lakes_stdev_moni_path).rename(columns={'indicator': 'parameter'})

lake_stdev0 = lake_stdev0[lake_stdev0.parameter.isin(param_mapping.keys())].replace({'parameter': param_mapping}).copy()
nan_errors = lake_stdev0[lake_stdev0.stdev.isnull()].LFENZID.values.astype(int)

lake_stdev0.loc[lake_stdev0.stdev <= list1[0], 'stdev'] = list1[0] *1.1
lake_stdev0.loc[lake_stdev0.stdev > list1[-1], 'stdev'] = list1[-1]
lake_stdev0.loc[lake_stdev0.stdev.isnull(), 'stdev'] = list1[-1]

errors1 = (pd.cut(lake_stdev0.stdev, list1, labels=list1[:-1]).to_numpy() * 1000).astype(int)
lake_stdev0['error'] = errors1

lakes_nps0 = pd.read_csv(lakes_nps_path)
lakes_nps0['conc_perc'] = ((1 - lakes_nps0['perc_reduction'].round(2)) * 100).astype('int8')

lake_stdev1 = pd.merge(lake_stdev0, lakes_nps0[['lawa_id', 'parameter', 'conc_perc']], on=['lawa_id', 'parameter'])

lake_sims = xr.open_dataset(lakes_sims_h5_path, engine='h5netcdf')

lake_sims1 = lake_sims.sel(n_samples=n_samples).to_dataframe().reset_index()

power_data0 = pd.merge(lake_stdev1, lake_sims1, on=['conc_perc', 'error']).drop('error', axis=1)
power_data0['conc_perc'] = 100 - power_data0['conc_perc']
power_data0 = power_data0.rename(columns={'conc_perc': 'reduction'})
power_data0['stdev'] = power_data0['stdev'].round(3)

power_data1 = power_data0[power_data0.reduction > 0]

power_data2 = pd.merge(n_samples_df, power_data1, on='n_samples').drop('n_samples', axis=1)

power_data2.set_index(['parameter', 'lawa_id', 'site_id', 'LFENZID', 'reduction', 'n_years', 'n_samples_year']).to_csv(base_path.joinpath(output_csv))

### Estimate number of samples
# n_years = [5, 10, 20, 30]
# n_samples_year = [26]

# n_samples0 = hdf5tools.utils.cartesian([n_years, n_samples_year])
# n_samples = np.prod(n_samples0, axis=1)
# n_samples.sort()

lake_sims2 = lake_sims.where(lake_sims.power >= 80)

lake_stdev2 = lake_stdev1[lake_stdev1.conc_perc < 100].copy()

r1 = pd.merge(lake_stdev2, lake_sims2.to_dataframe().reset_index().dropna(), on=['conc_perc', 'error']).drop('error', axis=1)
r1['conc_perc'] = 100 - r1['conc_perc']
r1 = r1.rename(columns={'conc_perc': 'reduction'})
r1['stdev'] = r1['stdev'].round(3)
r1['power'] = r1['power'].astype('int8')

r2 = pd.merge(n_samples_df, r1, on='n_samples').drop('n_samples', axis=1)

r2.set_index(['parameter', 'lawa_id', 'site_id', 'LFENZID', 'reduction', 'n_years', 'n_samples_year']).to_csv(base_path.joinpath(output2_csv))



































