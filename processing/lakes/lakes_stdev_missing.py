#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:19:12 2023

@author: mike
"""
import sys
import os
import pandas as pd
import numpy as np
import hdf5tools
import xarray as xr
import geopandas as gpd

if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

##################################################
### Preprocessing

stdev0 = pd.read_csv(utils.lakes_rupesh_stdev_path)
stdev0['LFENZID'] = stdev0['LFENZID'].astype('int32')

## Check that all 3rd order and greater are available
lakes0 = gpd.read_feather(utils.lakes_poly_path)

missing1s = lakes0[~lakes0.LFENZID.isin(stdev0.LFENZID)]

missing1s.to_file(utils.lakes_missing_3rd_path)

## Convert to log scale
stdev1 = stdev0.set_index('LFENZID').apply(lambda x: np.log(x), axis=1)































































