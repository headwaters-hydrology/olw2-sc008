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


def regress(results, data_col, date_col='date'):
    """

    """
    grp2 = results.groupby(['parameter', 'site_id'])

    r_list = []
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

        df1 = pd.DataFrame(zip(data[date_col].values, new_y), columns=['date', data_col])
        df1['parameter'] = i[0]
        df1['site_id'] = i[1]
        r_list.append(df1)

    df2 = pd.concat(r_list).set_index(['parameter', 'site_id', 'date'])

    return df2


def deseason(data):
    """

    """
    grp1 = data.groupby(['site_id', 'parameter'])[['date', 'observed']]

    raw_output_list = []
    for i, data in grp1:
        d1 = data.set_index('date')['observed']
        # d1 = d1[d1 != 0]
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
        s2 = pd.concat([d1, s1]).sort_index()
        # s2 = pd.concat([np.log(d1), s1]).sort_index()
        s3 = s2.interpolate('time')
        s4 = (s3 + s3.shift(-1))/2
        s5 = s4.resample(freq_code).mean().dropna()
        # s5 = s3[reg1]
        s5.name = 'observed'

        r1 = seasonal.STL(s5, robust=False, seasonal=13).fit()
        r2 = pd.concat([r1.observed, r1.trend, r1.seasonal, r1.resid], axis=1)
        r2.index.name = 'date'

        # Put the observations back in and use a linear interp to get the others
        # r2b = pd.concat([np.log(d1).to_frame(), r2]).sort_index()
        r2b = pd.concat([d1.to_frame(), r2]).sort_index()
        r2b = r2b[~r2b.index.duplicated(keep='last')].copy()
        r2b[['trend', 'season']] = r2b[['trend', 'season']].interpolate('time', limit_area='inside')
        r2b['resid'] = r2b['observed'] - r2b['trend'] - r2b['season']
        r2d = r2b.loc[d1.index, :].dropna()

        r2d['site_id'] = i[0]
        r2d['parameter'] = i[1]
        r3 = r2d.reset_index().set_index(['parameter', 'site_id', 'date'])
        raw_output_list.append(r3)

    all_results = pd.concat(raw_output_list)

    return all_results


def test_deseason_resampling(data):
    """

    """
    grp1 = data.groupby(['site_id', 'parameter'])[['date', 'observed']]

    output_list = []
    for i, data in grp1:
        d1 = data.set_index('date')['observed']
        d2 = d1.reset_index()['date'].shift(1) - d1.index
        m_days = -d2.dt.days.median()
        if m_days < 90:
            train = d1[::2]
            test = d1[1::2]
            d2 = train.reset_index()['date'].shift(1) - train.index
            m_days = -d2.dt.days.median()
            if m_days < 60:
                freq_code = '1M'
            elif m_days < 90:
                freq_code = '2M'
            else:
                freq_code = 'Q'

            ## Use full dataset at lower freq
            reg1 = pd.date_range(d1.index[0], d1.index[-1], freq=freq_code)
            reg2 = reg1[~reg1.isin(d1.index)]
            s1 = pd.Series(np.nan, index=reg2)
            s2 = pd.concat([d1, s1]).sort_index()
            s3 = s2.interpolate('time')
            s4 = (s3 + s3.shift(-1))/2
            s5 = s4.resample(freq_code).mean().dropna()
            s5.name = 'observed'

            r1 = seasonal.STL(s5, robust=False, seasonal=13).fit()
            r2 = pd.concat([r1.observed, r1.trend, r1.seasonal, r1.resid], axis=1)
            r2.index.name = 'date'

            # Put the observations back in and use a linear interp to get the others
            r2b = pd.concat([test.to_frame(), r2]).sort_index()
            r2b = r2b[~r2b.index.duplicated(keep='last')].copy()
            r2b[['trend', 'season']] = r2b[['trend', 'season']].interpolate('time', limit_area='inside')
            r2b['resid'] = r2b['observed'] - r2b['trend'] - r2b['season']
            r2d = r2b.loc[test.index, :].dropna()
            full1 = r2d['trend'] + r2d['resid']
            # base_stdev = r2e.std()

            ## Test by removing half the data
            reg1 = pd.date_range(train.index[0], train.index[-1], freq=freq_code)
            reg2 = reg1[~reg1.isin(train.index)]
            s1 = pd.Series(np.nan, index=reg2)
            s2 = pd.concat([train, s1]).sort_index()
            s3 = s2.interpolate('time')
            s4 = (s3 + s3.shift(-1))/2
            s5 = s4.resample(freq_code).mean().dropna()
            s5.name = 'observed'

            r1 = seasonal.STL(s5, robust=False, seasonal=13).fit()
            r2 = pd.concat([r1.observed, r1.trend, r1.seasonal, r1.resid], axis=1)
            r2.index.name = 'date'

            # Put the observations back in and use a linear interp to get the others
            r2b = pd.concat([test.to_frame(), r2]).sort_index()
            r2b = r2b[~r2b.index.duplicated(keep='last')].copy()
            r2b[['trend', 'season']] = r2b[['trend', 'season']].interpolate('time', limit_area='inside')
            r2b['resid'] = r2b['observed'] - r2b['trend'] - r2b['season']
            r2d = r2b.loc[test.index, :].dropna()
            test1 = r2d['trend'] + r2d['resid']
            # test_stdev = r2e.std()

            ## Combine results
            combo3 = pd.concat([full1, test1], axis=1).dropna()
            base_stdev, test_stdev = combo3.std()
            r3 = [i[1], i[0], freq_code, base_stdev, test_stdev]
            output_list.append(r3)

    all_results = pd.DataFrame(output_list, columns=['parameter', 'site_id', 'freq', 'base_stdev', 'test_stdev'])
    all_results['error'] = (all_results['test_stdev'] / all_results['base_stdev']) - 1
    summ_results1 = all_results.groupby(['parameter', 'freq']).mean(numeric_only=True)
    summ_results2 = all_results.groupby(['parameter', 'freq'])['error'].count()
    summ_results2.name = 'site_count'
    summ_results = pd.concat([summ_results1, summ_results2], axis=1)

    return all_results.set_index(['parameter', 'site_id', 'freq']), summ_results


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

## Deseasonalize the data - test removing overall trend before/after deseason
data0a = np.log(data0.observed[data0.observed != 0]) # log transformed

## Testing interp of the deseasonalize method
interp_results, interp_summ = test_deseason_resampling(data0a.reset_index())

interp_results.round(4).to_csv(utils.lakes_interp_test_results_path)
interp_summ.round(4).to_csv(utils.lakes_interp_test_summ_path)


## Deseasonalize the data - test removing overall trend before/after deseason
# data0a = data0.observed.copy() # Not log transformed
data0b = regress(data0a.reset_index(), date_col='date', data_col='observed')

all_results1 = deseason(data0b.reset_index())
all_results2 = deseason(data0a.reset_index())

results1 = all_results1['trend'] + all_results1['resid']
results1.name = 'no_trend_deseasoned'
results2a = all_results2['trend'] + all_results2['resid']
results2a.name = 'with_trend_deseasoned'

results2 = regress(results2a.reset_index(), date_col='date', data_col='with_trend_deseasoned')

combo1 = pd.merge(results1, results2, on=['parameter', 'site_id', 'date'])
combo2 = combo1.groupby(['parameter', 'site_id']).std()

diff1 = 1 - (combo2['no_trend_deseasoned'] / combo2['with_trend_deseasoned'])
print(diff1.abs().mean())
print(diff1.mean())

combo2.round(4).to_csv(utils.lakes_trend_comp_path)

## Use no trend deseasoned data
all_results1.round(4).to_csv(utils.lakes_deseason_path)

stdev0 = combo2['no_trend_deseasoned']
stdev0.name = 'deseasoned_stdev'

## Compare to non-deseasonalised results
obs_stdev0 = data0a.groupby(['parameter', 'site_id']).std()
obs_stdev0.name = 'observed_stdev'

combo0 = pd.merge(obs_stdev0, stdev0, on=['parameter', 'site_id'])

combo0.round(4).to_csv(utils.lakes_deseason_comp_path)

combo0['ratio'] = combo0['deseasoned_stdev']/combo0['observed_stdev']

combo1 = combo0.groupby('parameter')['ratio'].mean()

# The mean ratio for all parameters is 0.83. It varies little between parameters.

## Add in the LFENZIDs
wq_data = pd.read_csv(utils.lakes_data_path)
site_data = wq_data.rename(columns={'LawaSiteID': 'lawa_id', 'SiteID': 'site_id'})[['lawa_id', 'site_id', 'LFENZID']].dropna().drop_duplicates(subset=['lawa_id'])
site_data['LFENZID'] = site_data['LFENZID'].astype(int)
site_data = site_data[site_data.LFENZID > 0].copy()

stdev0.name = 'stdev'
stdev1 = stdev0.reset_index().rename(columns={'site_id': 'lawa_id'})
stdev_df1 = pd.merge(site_data, stdev1, on='lawa_id').rename(columns={'parameter': 'indicator'})

stdev_df1.to_csv(utils.lakes_stdev_moni_path, index=False)









































