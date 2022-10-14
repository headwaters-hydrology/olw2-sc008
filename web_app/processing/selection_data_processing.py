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

pd.options.display.max_columns = 10

##############################################
### Parameters

base_path = '/home/mike/data/OLW/web_app'
# %cd '/home/mike/data/OLW/web_app'

test_data_csv = 'NH_Results.csv'

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

data0 = pd.read_csv(test_data_csv, usecols=list(all_cols))
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















































































