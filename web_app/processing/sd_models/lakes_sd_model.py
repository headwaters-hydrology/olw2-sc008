#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:53:39 2023

@author: mike
"""
import os
import sys
import pandas as pd
import numpy as np
# from sklearn.preprocessing import PowerTransformer
from sklearn import linear_model
import scipy


if '..' not in sys.path:
    sys.path.append('..')

import utils


pd.options.display.max_columns = 10

################################################
### Functions


def reg_transform(x, y, slope, intercept):
    """

    """
    y_new = y - (slope*x + intercept)
    return y_new



################################################
### Individual SD from regressions

data0 = pd.read_feather(utils.lakes_data_clean_path)

grp1 = data0.groupby(['indicator'])['value']
for i, v in grp1:
    new_data, l = scipy.stats.boxcox(v)
    data0.loc[data0.indicator == i, 'value'] = new_data


grp2 = data0.groupby(['LawaSiteID', 'indicator'])[['date', 'value']]

stdev_list = []
for i, v in grp2:
    x = v.date.values.astype('datetime64[D]').astype(int)
    y = v.value
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

stdev_df = pd.DataFrame(stdev_list, columns=['LawaSiteID', 'indicator', 'stdev'])

stdev_df1 = pd.merge(data0[['LawaSiteID', 'LFENZID']].drop_duplicates(), stdev_df, on=['LawaSiteID'])

stdev_df2 = stdev_df1.groupby(['LFENZID', 'indicator'])['stdev'].mean()

































































































