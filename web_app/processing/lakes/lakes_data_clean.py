#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:53:57 2023

@author: mike
"""
import os
import pandas as pd
import numpy as np

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils


pd.options.display.max_columns = 10


#############################################################
### Parameters


##############################################################
### Process data

data0 = pd.read_csv(utils.lakes_data_path)

# data0['SampleDate'] = pd.to_datetime(data0['SampleDate'], infer_datetime_format=True)
data0['SampleDate'] = pd.to_datetime(data0['SampleDate'], format='%d-%b-%y')

data1 = data0[data0['Symbol'] != 'Right'].copy()
data1['censored'] = True
data1.loc[data1['Symbol'] == '0', 'censored'] = False

grp1 = data1.groupby(['LawaSiteID', 'Indicator'])['censored']

sum_censored = grp1.sum()
n_censored = grp1.count()
bad_combos = ((sum_censored / n_censored) > 0.33) | (n_censored < 10)

good_ones = bad_combos[bad_combos == False].reset_index().drop('censored', axis=1)

data2 = pd.merge(good_ones, data1, on=['LawaSiteID', 'Indicator'])

data2.loc[data2['censored'], 'CensoredValue'] = data2.loc[data2['censored'], 'CensoredValue'] * 0.5

data2 = data2.drop(['Symbol', 'Units', 'censored'], axis=1)
data2 = data2[data2.CensoredValue > 0].copy()

data3 = data2.rename(columns={'SampleDate': 'date', 'CensoredValue': 'value', 'Indicator': 'indicator'}).dropna()
data3['LFENZID'] = data3['LFENZID'].astype('int32')

grp1 = data3.groupby(['LawaSiteID', 'indicator', 'date'])

v1 = grp1['value'].mean()
c1 = grp1[['SiteID', 'LFENZID']].first()

data4 = pd.concat([c1, v1], axis=1)

data4.reset_index().to_feather(utils.lakes_data_clean_path, compression='zstd', compression_level=1)








































