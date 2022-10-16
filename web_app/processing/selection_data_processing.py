#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:27:35 2022

@author: mike
"""
import os
import pandas as pd
import numpy as np
import pickle
import io
import pathlib
import hdf5tools
import hdf5plugin
import xarray as xr
import copy
from geopandas import gpd
from sklearn.neighbors import KernelDensity

pd.options.display.max_columns = 10

##############################################
### Parameters

base_path = '/home/mike/data/olw/web_app'
# %cd '/home/mike/data/OLW/web_app'

base_path = pathlib.Path(base_path)

test_data_csv = 'NH_Results.csv'

rec_streams_gpkg = 'rec_streams3plus.gpkg'

index_cols = ['indicator', 'nzsegment', 'percent_change', 'time_period', 'frequency']
rename_cols = {'Indicator': 'indicator', 'Percent_Change': 'percent_change', 'Time_Period': 'time_period', 'Frequency': 'frequency', 'NHlmpwr': 'percent_likelihood'}

all_cols = set(['nzsegment'])
all_cols.update(rename_cols.keys())

dtype_dict = {'nzsegment': 'int32', 'indicator': str, 'percent_change': 'int8', 'time_period': 'int8', 'frequency': 'int16', 'percent_likelihood': 'float32'}

encoding = {'percent_likelihood': {'dtype': 'int8', 'missing_value': -99}}

#############################################
### Functions




############################################
### Processing

data0 = pd.read_csv(base_path.joinpath(test_data_csv), usecols=list(all_cols))
data1 = data0.rename(columns=rename_cols).drop_duplicates(index_cols)

# data1.loc[data1['percent_likelihood'].isnull(), 'percent_likelihood'] = 0
data1['percent_likelihood'] = data1['percent_likelihood'].round()
data1['time_period'] = data1['time_period'].round()
data1['frequency'] = data1['frequency'].round()
data1['percent_change'] = (data1['percent_change']*100).round()

data2 = data1.set_index(index_cols).to_xarray()

for var in data2.variables:
    if var in dtype_dict:
        data2[var] = data2[var].astype(dtype_dict[var])
    if var in encoding:
        data2[var].encoding = encoding[var]

## Set chunking
dims = data2['percent_likelihood'].dims
chunks = {'percent_likelihood': list(data2['percent_likelihood'].shape)}
chunks['percent_likelihood'][dims.index('indicator')] = 1
if chunks['percent_likelihood'][dims.index('nzsegment')] > 15000:
    chunks['percent_likelihood'][dims.index('nzsegment')] = 15000

chunks['percent_likelihood'] = tuple(chunks['percent_likelihood'])

## Save
hdf5tools.xr_to_hdf5(data2, os.path.join(base_path, 'selection_data.h5'), chunks=chunks)



###########################################
### Create data from the sample data

rec_streams = gpd.read_file(base_path.joinpath(rec_streams_gpkg), driver='GPKG')

kde = KernelDensity(kernel='gaussian', bandwidth=0.2)

grp1 = data1.groupby(['percent_change', 'time_period', 'frequency'])['percent_likelihood']

data4 = data2.drop('percent_likelihood').copy()
nzseg = rec_streams['nzsegment'].values.astype('int32')
nzseg.sort()
data4['nzsegment'] = nzseg
data4['indicator'] = ['NH4', 'NO3', 'PO4']
data4['percent_likelihood'] = (tuple(data4.dims), np.zeros(tuple(data4.dims.values()), 'int8'))

min1 = data4['nzsegment'].values.min()
max1 = data4['nzsegment'].values.max()
len1 = len(data4['nzsegment'].values)

for ind in data4['indicator']:
    for g, val in grp1:
        g1 = [int(i) for i in g]
        g1.insert(0, ind)
        g1.insert(1, slice(min1, max1))
        arr = val.dropna().values
        fitted = kde.fit(arr.reshape(758, 1))
        # score = kde.score(arr.reshape(758, 1))
        output = fitted.sample(len1).flatten().round()
        output[output > 100] = 100
        output[output < 0] = 0
        data4['percent_likelihood'].loc[tuple(g1)] = output.astype('int8')

## Save
chunks = {'percent_likelihood': (1, 10000, 4, 4, 5)}
hdf5tools.xr_to_hdf5(data4, os.path.join(base_path, 'selection_data_all.h5'), chunks=chunks)






































































