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

base_path = pathlib.Path('/media/nvme1/data/OLW')

rivers_reach_error_path = '/media/nvme1/data/OLW/web_app/output/rivers_reaches_error_gam.h5'

sites_csv = 'olw_river_sites_rec.csv'

output_csv = 'power_calcs_above_3rd_order_gam_v02.csv'

reductions = [20, 30]

n_years = [5, 20]
n_samples_year = [12, 26, 52, 104, 364]

n_samples0 = hdf5tools.utils.cartesian([n_years, n_samples_year])
n_samples = np.prod(n_samples0, axis=1)

n_samples_df = pd.DataFrame(n_samples0, columns=['n_years', 'n_samples_year'])
n_samples_df['n_samples'] = n_samples

#####################################################
### Process data

w0 = nzrec.Water('/media/nvme1/git/nzrec/data')

stream_orders = {way_id: v['Strahler stream order'] for way_id, v in w0._way_tag.items()}

ways_4th_up = set([i for i, v in stream_orders.items() if v > 3])

conc0 = pd.read_csv('/media/nvme1/data/OLW/web_app/StBD3.csv', usecols=['Indicator', 'nzsegment', 'lm1seRes']).dropna()
nzsegments = conc0.nzsegment.unique().astype('int32')

power_data = xr.open_dataset(rivers_reach_error_path, engine='h5netcdf')

power_data['nzsegment'] = power_data['nzsegment'].astype('int32')
power_data['n_samples'] = power_data['n_samples'].astype('int16')
power_data['conc_perc'] = power_data['conc_perc'].astype('int8')

segs = power_data.nzsegment.values[np.in1d(power_data.nzsegment.values, nzsegments)]
segs = segs[np.in1d(segs, np.asarray(list(ways_4th_up), dtype='int32'))]

power_data = power_data.sel(conc_perc=[100 - red for red in reductions], n_samples=n_samples, nzsegment=segs).load()
power_data['conc_perc'] = reductions
power_data = power_data.rename({'conc_perc': 'reduction'})

power_data['reduction'] = power_data['reduction'].astype('int8')
power_data['power'] = power_data['power'].astype('int8')

pd_df1 = power_data.to_dataframe().reset_index()
pd_df1 = pd.merge(n_samples_df, pd_df1, on='n_samples').drop('n_samples', axis=1)

pd_df1[['nzsegment', 'indicator', 'reduction', 'n_years', 'n_samples_year', 'power']].sort_values(['nzsegment', 'indicator', 'reduction', 'n_years', 'n_samples_year']).to_csv(base_path.joinpath(output_csv), index=False)


