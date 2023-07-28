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


def stdev_regress(results, date_col='date', data_col='deseasoned'):
    """

    """
    grp2 = results.groupby(['parameter', 'site_id'])

    stdev_list = []
    for i, data in grp2:
        x = data[date_col].values.astype('datetime64[D]').astype(int)
        y = data[data_col].values
        slope, intercept, r, p, se = scipy.stats.linregress(x, y)
        if p < 0.05:
            new_y = []
            for x1, y1 in zip(x, y):
                new_y.append(reg_transform(x1, y1, slope, intercept))
            new_y = np.array(new_y)
        else:
            new_y = y - y.mean()

        stdev = new_y.std()
        stdev_list.append([i[0], i[1], stdev])

    stdev_df = pd.DataFrame(stdev_list, columns=['parameter', 'lawa_id', 'stdev'])

    return stdev_df

##############################################################
### Process data

data_list = []
for param in params:
    dir_name = base_dir_name.format(param=param)
    param_path = utils.lakes_source_path.joinpath(dir_name)
    for path in param_path.iterdir():
        # site_name = path.name.split(base_file_name)[1].split('.')[0]
        data = pd.read_csv(path).iloc[:, 1:].rename(columns={'ID': 'site_id', 'Date': 'date', 'Observed': 'observed'})
        data['date'] = pd.to_datetime(data['date'], dayfirst=True)
        data['parameter'] = param
        data_list.append(data)

data0 = pd.concat(data_list).set_index(['site_id', 'parameter', 'date']).sort_index()
data0.to_csv(utils.lakes_source_data_path)

## Deseasonalize the data
data1 = data0.reset_index()
grp1 = data1.groupby(['site_id', 'parameter'])[['date', 'observed']]

raw_output_list = []
results_list = []
for i, data in grp1:
    d1 = data.set_index('date')['observed']
    d1 = d1[d1 != 0]
    d2 = d1.reset_index()['date'].shift(1) - d1.index
    m_days = -d2.dt.days.median()
    if m_days < 60:
        freq_code = 'M'
        # seasonal = 13
    elif m_days < 90:
        freq_code = '2M'
        # seasonal = 7
    else:
        freq_code = 'Q'
    reg1 = pd.date_range(d1.index[0], d1.index[-1], freq=freq_code)
    reg2 = reg1[~reg1.isin(d1.index)]
    s1 = pd.Series(np.nan, index=reg2)
    # s2 = pd.concat([d1, s1]).sort_index()
    s2 = pd.concat([np.log(d1), s1]).sort_index()
    s3 = s2.interpolate('time')
    s4 = (s3 + s3.shift(-1))/2
    s5 = s4.resample(freq_code).mean().dropna()
    # s5 = s3[reg1]
    s5.name = 'observed'

    r1 = seasonal.STL(s5, robust=False, seasonal=13).fit()
    r2 = pd.concat([r1.observed, r1.trend, r1.seasonal, r1.resid], axis=1)
    r2.index.name = 'date'
    r2['site_id'] = i[0]
    r2['parameter'] = i[1]
    r3 = r2.reset_index().set_index(['parameter', 'site_id', 'date'])
    combo0 = r3.trend + r3.resid
    combo0.name = 'deseasoned'
    raw_output_list.append(r3)
    results_list.append(combo0)

all_results = pd.concat(raw_output_list)
results = pd.concat(results_list)

all_results.to_csv(utils.lakes_deseason_path)


## De-trend and calc stdev
stdev_df = stdev_regress(results.reset_index(), date_col='date', data_col='deseasoned')

## Compare to non-deseasonalised results
stdev0 = stdev_regress(all_results.reset_index(), date_col='date', data_col='observed').rename(columns={'stdev': 'stdev_base'})

combo0 = pd.merge(stdev_df, stdev0, on=['parameter', 'lawa_id'])

combo0.to_csv(utils.lakes_deseason_comp_path, index=False)

combo0['ratio'] = combo0['stdev']/combo0['stdev_base']

combo1 = combo0.groupby('parameter')['ratio'].mean()

# The mean ratio for all parameters is 0.84. It varies little between parameters.

## Add in the LFENZIDs
wq_data = pd.read_csv(utils.lakes_data_path)
site_data = wq_data.rename(columns={'LawaSiteID': 'lawa_id', 'SiteID': 'site_id'})[['lawa_id', 'site_id', 'LFENZID']].dropna().drop_duplicates(subset=['lawa_id'])
site_data['LFENZID'] = site_data['LFENZID'].astype(int)
site_data = site_data[site_data.LFENZID > 0].copy()

stdev_df1 = pd.merge(site_data, stdev_df, on='lawa_id').rename(columns={'parameter': 'indicator'})

stdev_df1.to_csv(utils.lakes_stdev_moni_path, index=False)









































