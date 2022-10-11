#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:30:36 2022

@author: mike
"""
import os
from gistools import vector
import geopandas as gpd
import pandas as pd
import numpy as np

pd.options.display.max_columns = 10

##############################################
### Parameters

base_path = '/home/mike/data/OLW/web_app'

rec_rivers_shp = '/home/mike/data/niwa/rec/River_Lines.shp'
rec_catch_shp ='/home/mike/data/niwa/rec/ca240805-e6be-414e-8a39-579459acac2a2020230-1-1l4v6za.5259g.shp'

test_data_csv = 'NH_Results.csv'

#############################################
### Testing

rec_rivers0 = gpd.read_file(os.path.join(base_path, rec_rivers_shp))

test_data0 = pd.read_csv(os.path.join(base_path, test_data_csv))




























































































