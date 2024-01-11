#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:11:07 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import intersection
import hdf5tools
import xarray as xr
import dbm
import booklet
import pickle
import zstandard as zstd

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

##########################################
### land use/cover

# parcels = gpd.read_file(utils.parcels_path)
# parcels = parcels[['id', 'geometry']].copy()

lcdb_classes = ['Exotic Forest', 'Forest - Harvested', 'Orchard, Vineyard or Other Perennial Crop', 'Short-rotation Cropland', 'Built-up Area (settlement)', 'High Producing Exotic Grassland', 'Low Producing Grassland', 'Mixed Exotic Shrubland']

lcdb_where_sql = 'Name_2018 IN ({})'.format(str(lcdb_classes)[1:-1])


def lcdb_processing():
    ## land cover
    lcdb0 = gpd.read_file(utils.lcdb_path, include_fields=['Name_2018'], where=lcdb_where_sql)
    lcdb0['geometry'] = lcdb0.buffer(0).simplify(1)
    lcdb0 = lcdb0.rename(columns={'Name_2018': 'land_cover'})
    lcdb0['farm_type'] = 'NA'
    lcdb0['typology'] = lcdb0['land_cover']
    # lcdb1 = lcdb0.dissolve('land_cover')

    utils.gpd_to_feather(lcdb0, utils.lcdb_clean_path)



































































































