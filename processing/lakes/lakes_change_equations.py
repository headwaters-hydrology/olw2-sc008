#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:16:57 2023

@author: mike
"""
import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
# from sklearn.preprocessing import PowerTransformer
from sklearn import linear_model, kernel_ridge
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
import scipy
import hdf5tools
import math


if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

################################################
### Parameters

tn_conc = 840
tp_conc = 62.4
chla_conc = 23.7

residence_time = 10
max_depth = 20

conc_perc0 = np.arange(1, 100)
conc = 0.5

################################################
### Change ratios

## TP
b = (1 + (0.44*(residence_time**0.13)))
res_base = 10**(np.log10(tp_conc)/b)

results_list = []

for conc in conc_perc0:
    b = (1 + (0.44*(residence_time**0.13)))
    res_red = 10**(np.log10(tp_conc*(conc*0.01))/b)
    ratio = res_red/res_base
    results_list.append(ratio)


## TN
b = 39.8/(max_depth**0.41)
res_base = b*(tn_conc**0.54)

res_red = b*((tn_conc*(conc*0.01))**0.54)

ratio = res_red/res_base

## Chla
res_base = 10**(-1.8 + (0.7*np.log10((tn_conc/tp_conc)* tp_conc)) + 0.55*np.log10(tp_conc))

res_red = 10**(-1.8 + (0.7*np.log10((tn_conc/tp_conc) * tp_conc * 0.5)) + 0.55*np.log10(tp_conc * 0.5))

ratio = res_red/res_base

## Secchi
# Deep
res_base = (3.46 - 1.53*np.log10(chla_conc))**2

res_red = (3.46 - 1.53*np.log10(chla_conc*0.9))**2

ratio = res_red/res_base

results_dict = {}

for conc in conc_perc0:
    res_red = (3.46 - 1.53*np.log10(chla_conc*conc*0.01))**2
    ratio = 1/(res_red/res_base)
    results_dict[conc] = math.log(ratio, conc*0.01)


# Shallow
res_base = (3.46 - 0.74*np.log10(chla_conc) - 0.35*np.log10(200/10))**2

results_dict = {}

for conc in conc_perc0:
    res_red = (3.46 - 0.74*np.log10(chla_conc*conc*0.01) - 0.35*np.log10(200/10))**2
    ratio = 1/(res_red/res_base)
    results_dict[conc] = math.log(ratio, conc*0.01)







































































































