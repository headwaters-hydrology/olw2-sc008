#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:11:38 2023

@author: mike
"""
import pathlib
import os
import xarray as xr
import pandas as pd
import hdf5plugin

pd.options.display.max_columns = 10

############################################################
### Parameters

base_path = pathlib.Path('/media/nvme1/data/OLW/web_app/output/assets')

sims_h5 = 'rivers_sims.h5'

reaches_error_h5 = 'rivers_reaches_error.h5'

#############################################################
### extract data

reaches_error = xr.open_dataset(base_path.joinpath(reaches_error_h5), engine='h5netcdf')
sims = xr.open_dataset(base_path.joinpath(sims_h5), engine='h5netcdf')























































































