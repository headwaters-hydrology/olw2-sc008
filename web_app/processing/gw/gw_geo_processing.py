#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:17:00 2022

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
import booklet
import multiprocessing as mp
import concurrent.futures
import geobuf

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing


def gw_geo_process():

    rc0 = gpd.read_file(utils.rc_bounds_gpkg)
    rc0['name'] = rc0['REGC2023_V1_00_NAME_ASCII'].str[:-7]
    rc1 = rc0[rc0.name != 'Area Outside'][['name', 'geometry']].copy()
    rc1['geometry'] = rc1.buffer(0).simplify(1)
    rc2 = rc1.to_crs(4326)

    # Point locations
    gw_data = xr.open_dataset(utils.gw_monitoring_data_path)
    gw_pts0 = gw_data[['lon', 'lat', 'depth']].to_dataframe().reset_index()
    gw_data.close()

    gw_pts1 = vector.xy_to_gpd(['ref', 'depth'], 'lon', 'lat', gw_pts0, 4326)
    gw_pts1['geometry'] = gw_pts1['geometry'].simplify(0.00001)
    gw_pts1['tooltip'] = gw_pts1['ref']

    count = 0
    gw_dict = {}
    for name, poly in rc2.groupby('name'):
        gw_pts2 = gw_pts1[gw_pts1.within(poly['geometry'].iloc[0])]
        count += len(gw_pts2)
        sites_geo = gw_pts2.set_index('ref', drop=False).__geo_interface__
        sites_gbuf = geobuf.encode(sites_geo)
        gw_dict[name] = sites_gbuf

    print(count)
    # utils.gpd_to_feather(gw_pts1, utils.gw_points_path)

    with booklet.open(utils.gw_points_rc_blt, 'n', value_serializer=None, key_serializer='str', n_buckets=100) as gw:
        for name, gbuf in gw_dict.items():
            gw[name] = gbuf

    rc3 = rc1.copy()
    rc3['geometry'] = rc1.buffer(0).simplify(30)

    rc_geo = rc3.to_crs(4326).set_index('name', drop=False).__geo_interface__
    rc_gbuf = geobuf.encode(rc_geo)

    with open(utils.rc_bounds_gbuf, 'wb') as f:
        f.write(rc_gbuf)






















































