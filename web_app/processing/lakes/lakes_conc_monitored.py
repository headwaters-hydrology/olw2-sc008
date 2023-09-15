#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:53:57 2023

@author: mike
"""
import os
import pandas as pd
import numpy as np
from statsmodels.tsa import seasonal
import scipy

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils


pd.options.display.max_columns = 10


#############################################################
### Parameters

params = ['TN', 'Secchi', 'TP', 'Chla']
base_dir_name = '{param}_Decomposed_Timeseries'
base_file_name = 'Decompose'
freq_code = 'Q'



#############################################################
### Functions


def reg_transform(x, y, slope, intercept):
    """

    """
    y_new = y - (slope*x + intercept)
    return y_new


def est_median_freq(data):
    """

    """
    d2 = data['date'].shift(1) - data['date']
    m_days = -d2.dt.days.median()

    return m_days


def trend_values(results, data_col, data_date, date_col='date'):
    """

    """
    data_date_int = np.datetime64(data_date, 'D').astype(int)
    year = data_date[:4]
    grp2 = results.groupby(['parameter', 'lawa_id'])

    r_list = []
    for i, data in grp2:
        x = data[date_col].values.astype('datetime64[D]').astype(int)
        y = data[data_col].values
        slope, intercept, r, p, se = scipy.stats.linregress(x, y)
        new_val = slope*data_date_int + intercept

        r_list.append([i[0], i[1], new_val])

    df1 = pd.DataFrame(r_list, columns=['parameter', 'lawa_id', f'conc_{year}'])

    return df1


##############################################################
### Process data

## monitoring data from individual files
# data_list = []
# for param in params:
#     dir_name = base_dir_name.format(param=param)
#     param_path = utils.lakes_source_path.joinpath(dir_name)
#     for path in param_path.iterdir():
#         # site_name = path.name.split(base_file_name)[1].split('.')[0]
#         data = pd.read_csv(path).iloc[:, 1:].rename(columns={'ID': 'site_id', 'Date': 'date', 'Observed': 'observed'})
#         data['date'] = pd.to_datetime(data['date'], dayfirst=True)
#         data['parameter'] = param
#         data_list.append(data)

# data0 = pd.concat(data_list).set_index(['site_id', 'parameter', 'date']).sort_index()
# data0.to_csv(utils.lakes_source_data_path)

## monitoring data from stdev processing
data0 = pd.read_csv(utils.lakes_filtered_data_path)
data0['date'] = pd.to_datetime(data0['date'])
data0 = data0.set_index(['parameter', 'lawa_id', 'date']).observed

data0a = np.log(data0) # log transformed

## All the annual stats
grp1 = data0a.reset_index().groupby(['parameter', 'lawa_id'])

results_list = []
for i, data in grp1:
    param = i[0]
    lawa_id = i[1]
    d1 = data.set_index('date')['observed'].resample('A')

    if param == 'TP':
        d2 = d1.median().interpolate('time')
        d3 = d2.reset_index()
        d3['parameter'] = 'Total phosphorus median'
    elif param == 'TN':
        d2 = d1.median().interpolate('time')
        d3 = d2.reset_index()
        d3['parameter'] = 'Total nitrogen median'
    elif param == 'NH4N':
        d2 = d1.median().interpolate('time')
        d3 = d2.reset_index()
        d3['parameter'] = 'Ammoniacal nitrogen median'
    elif param == 'CHLA':
        d2 = d1.median().interpolate('time')
        d3 = d2.reset_index()
        d3['parameter'] = 'Chla median'

    d3['lawa_id'] = lawa_id

    results_list.append(d3)

data0b = pd.concat(results_list)

## Select trend end values
# results_list = []
# for date in ['2020-12-31', '2025-12-31', '2040-12-31']:
#     year = date[:4]
#     data1a = trend_values(data0b, 'observed', date, 'date')
#     data1a[f'conc_{year}'] = np.exp(data1a[f'conc_{year}'])
#     results_list.append(data1a.set_index(['parameter', 'lawa_id']))

# data1 = pd.concat(results_list, axis=1)

data1 = trend_values(data0b, 'observed', '2020-12-31', 'date')
data1['conc_2020'] = np.exp(data1['conc_2020'])

## Add in the LFENZIDs
wq_data = pd.read_csv(utils.lakes_data_path)
site_data = wq_data.rename(columns={'LawaSiteID': 'lawa_id', 'SiteID': 'site_id'})[['lawa_id', 'site_id', 'LFENZID']].dropna().drop_duplicates(subset=['lawa_id'])
site_data['LFENZID'] = site_data['LFENZID'].astype(int)
site_data = site_data[site_data.LFENZID > 0].copy()

df1 = pd.merge(site_data, data1, on='lawa_id').rename(columns={'parameter': 'indicator'})

df1.to_csv(utils.lakes_conc_moni_path, index=False)









































