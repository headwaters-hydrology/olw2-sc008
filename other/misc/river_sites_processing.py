#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:52:39 2023

@author: mike
"""
import os
import pandas as pd
import pathlib
import geopandas as gpd
from gistools import vector

pd.options.display.max_columns = 10

######################################################
### Parameters

base_path = pathlib.Path('/media/nvme1/data/OLW')

sites_loc_csv = 'olw_river_sites_locations.csv'
sites_rec_csv = 'olw_river_sites_rec.csv'

output_file = 'olw_river_sites.feather'

#####################################################
### Process data

sites_loc0 = pd.read_csv(base_path.joinpath(sites_loc_csv))

sites_loc1 = vector.xy_to_gpd('site_id', 'NZTMX', 'NZTMY', sites_loc0, 2193)

sites_rec0 = pd.read_csv(base_path.joinpath(sites_rec_csv)).dropna()
sites_rec0['nzsegment'] = sites_rec0['nzsegment'].astype('int32')

sites0 = sites_loc1.merge(sites_rec0, on='site_id').reset_index(drop=True)

sites0.to_feather(base_path.joinpath(output_file))











