#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:10:51 2023

@author: mike
"""
import os
import pandas as pd
import pathlib
import numpy as np
from scipy import stats

pd.options.display.max_columns = 10


##############################################
### Parameters

base_path = pathlib.Path('/media/nvme1/data/OLW/lakes/v05')

params = ['tn', 'tp', 'chla', 'secchi']

#############################################
### Processing

data_list = []
for param in params:
    param_path = base_path.joinpath(param)
    for file_path in param_path.iterdir():
        data0 = pd.read_csv(file_path).dropna()
        site = data0.iloc[0]['ID']
        residual_sd = data0['random'].std()
        combo0 = data0['random'] + data0['trend']
        observed0 = data0['Observed']
        time0 = pd.to_datetime(data0['Date'], dayfirst=True, infer_datetime_format=True).values.astype('datetime64[D]').astype(int)

        mean_conc = observed0.mean()

        ## Regressions
        observed1 = stats.linregress(time0, observed0)
        slope1 = (time0*observed1.slope) + observed1.intercept
        observed_residuals = (observed0 - slope1).std()

        combo1 = stats.linregress(time0, combo0)
        slope1 = (time0*observed1.slope) + observed1.intercept
        combo_residuals = (combo0 - slope1).std()

        ## Results
        data_list.append([param, site, mean_conc, observed0.median(), observed_residuals, observed_residuals/mean_conc, combo_residuals, combo_residuals/mean_conc, residual_sd, residual_sd/mean_conc])

results0 = pd.DataFrame(data_list, columns=['parameter', 'lawa_id', 'mean_conc', 'median_conc', 'observed_reg_sd', 'observed_reg_sd/mean', 'deseasonalised_reg_sd', 'deseasonalised_reg_sd/mean', 'deseasonalised_residuals', 'deseasonalised_residuals/mean'])

































































































































