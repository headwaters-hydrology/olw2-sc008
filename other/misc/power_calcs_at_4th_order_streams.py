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

rivers_reach_error_path = '/home/mike/data/OLW/web_app/output/assets/rivers_reaches_power_modelled.h5'

sites_csv = 'lawa_to_nzsegment.csv'

output_csv = 'power_calcs_above_3rd_order_gam_v04.csv'

reductions = [30]

n_years = [5, 20]
n_samples_year = [12, 26, 52, 104, 364]

n_samples0 = hdf5tools.utils.cartesian([n_years, n_samples_year])
n_samples = np.prod(n_samples0, axis=1)

n_samples_df = pd.DataFrame(n_samples0, columns=['n_years', 'n_samples_year'])
n_samples_df['n_samples'] = n_samples

#####################################################
### Process data

w0 = nzrec.Water('/home/mike/git/nzrec/data')

stream_orders = {way_id: v['Strahler stream order'] for way_id, v in w0._way_tag.items()}

ways_4th_up = [i for i, v in stream_orders.items() if v > 3]

conc0 = pd.read_csv('/home/mike/data/OLW/web_app/rivers/StBD3.csv', usecols=['Indicator', 'nzsegment', 'lm1seRes']).dropna()
nzsegments = conc0.nzsegment.unique().astype('int32')

sites = pd.read_csv(base_path.joinpath(sites_csv)).dropna()
sites['nzsegment'] = sites['nzsegment'].astype('int32')

power_data = xr.open_dataset(rivers_reach_error_path, engine='h5netcdf')

power_data['nzsegment'] = power_data['nzsegment'].astype('int32')
power_data['n_samples'] = power_data['n_samples'].astype('int16')
power_data['conc_perc'] = power_data['conc_perc'].astype('int8')

segs = power_data.nzsegment.values[np.in1d(power_data.nzsegment.values, nzsegments)]
segs = segs[np.in1d(segs, np.asarray(ways_4th_up, dtype='int32')) | np.in1d(segs, sites.nzsegment.unique())]

power_data = power_data.sel(conc_perc=[100 - red for red in reductions], n_samples=n_samples, nzsegment=segs).load()
power_data['conc_perc'] = reductions
power_data = power_data.rename({'conc_perc': 'reduction'})

power_data['reduction'] = power_data['reduction'].astype('int8')
power_data['power'] = power_data['power'].astype('int8')

pd_df1 = power_data.to_dataframe().reset_index()
pd_df1 = pd.merge(n_samples_df, pd_df1, on='n_samples').drop('n_samples', axis=1)

s_orders_df = pd.DataFrame.from_dict(stream_orders, orient='index', columns=['stream_order'])
s_orders_df.index.name = 'nzsegment'
s_orders_df = s_orders_df.reset_index()

pd_df1 = pd.merge(pd_df1, s_orders_df, on='nzsegment', how='left')
pd_df1 = pd.merge(pd_df1, sites, on='nzsegment', how='left')

pd_df1[['nzsegment', 'indicator', 'reduction', 'n_years', 'n_samples_year', 'power', 'stream_order', 'lawa_id']].sort_values(['nzsegment', 'indicator', 'reduction', 'n_years', 'n_samples_year']).to_csv(base_path.joinpath(output_csv), index=False)


