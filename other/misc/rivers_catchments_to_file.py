#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 08:41:14 2023

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from copy import copy
import nzrec
import booklet
import orjson

pd.options.display.max_columns = 10


#############################################
### Parameters

catch_blt_path = '/home/mike/data/OLW/web_app/output/assets/rivers_catchments_major.blt'
catch_gpkg_path = '/home/mike/data/OLW/temp/rivers_catchments_gte_3rd_order.gpkg'

#############################################
### Processing

geo_list = []
catch_ids_list = []
with booklet.open(catch_blt_path) as f:
    for catch_id, geo in f.items():
        catch_ids_list.append(catch_id)
        geo_list.append(geo)



rec_shed = gpd.GeoDataFrame(catch_ids_list, geometry=geo_list, crs=4326, columns=['nzsegment'])
# rec_shed['geometry'] = rec_shed.simplify(0.0004)

rec_shed.to_file(catch_gpkg_path)


























