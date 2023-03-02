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

pd.options.display.max_columns = 10

######################################################
### Parameters

base_path = pathlib.Path('/media/nvme1/data/OLW')

rivers_reach_error_path = '/media/nvme1/data/OLW/web_app/output/rivers_reaches_error_gam.h5'

sites_csv = 'olw_river_sites_rec.csv'

output_csv = 'power_calcs_at_river_sites_gam_v03.csv'

reductions = [10, 20, 30]

n_years = [5, 20]
n_samples_year = [12, 26, 52, 104, 364]

n_samples0 = hdf5tools.utils.cartesian([n_years, n_samples_year])
n_samples = np.prod(n_samples0, axis=1)

n_samples_df = pd.DataFrame(n_samples0, columns=['n_years', 'n_samples_year'])
n_samples_df['n_samples'] = n_samples

#####################################################
### Process data

power_data = xr.open_dataset(rivers_reach_error_path, engine='h5netcdf')

sites = pd.read_csv(base_path.joinpath(sites_csv)).dropna()

sites['nzsegment'] = sites['nzsegment'].astype('int32')
power_data['nzsegment'] = power_data['nzsegment'].astype('int32')
power_data['n_samples'] = power_data['n_samples'].astype('int16')
power_data['conc_perc'] = power_data['conc_perc'].astype('int8')

segs = power_data.nzsegment.values[np.in1d(power_data.nzsegment.values, sites['nzsegment'].unique())]

power_data = power_data.sel(conc_perc=[100 - red for red in reductions], n_samples=n_samples, nzsegment=segs).load()
power_data['conc_perc'] = reductions
power_data = power_data.rename({'conc_perc': 'reduction'})

power_data['reduction'] = power_data['reduction'].astype('int8')
power_data['power'] = power_data['power'].astype('int8')

pd_df1 = power_data.to_dataframe().reset_index()
pd_df1 = pd.merge(n_samples_df, pd_df1, on='n_samples').drop('n_samples', axis=1)

combo = pd.merge(sites, pd_df1, on='nzsegment')

combo[['site_id', 'nzsegment', 'indicator', 'reduction', 'n_years', 'n_samples_year', 'power']].sort_values(['nzsegment', 'indicator', 'reduction', 'n_years', 'n_samples_year']).to_csv(base_path.joinpath(output_csv), index=False)


