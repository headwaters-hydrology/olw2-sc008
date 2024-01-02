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
import scipy

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing

# lag_priorities = [
#     {'lag_dist': 500},
#     {'lag_dist': 500},


#     ]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]



def gw_geo_process():

    rc0 = gpd.read_file(utils.rc_bounds_gpkg)
    rc0['name'] = rc0['REGC2023_V1_00_NAME_ASCII'].str[:-7]
    rc1 = rc0[rc0.name != 'Area Outside'][['name', 'geometry']].copy()
    rc1['geometry'] = rc1.buffer(0).simplify(1)
    rc2 = rc1.to_crs(4326)

    # Point locations
    gw_data = xr.open_dataset(utils.gw_monitoring_data_path)

    # Determine the lags per ref
    lag_depths = gw_data.lag_depth.values
    lag_depths[-1] = 500
    gw_data['lag_depth'] = lag_depths.astype('int16')

    gw_data = gw_data.isel(lag_dist=range(8)).copy()
    gw_data['lag_dist'] = gw_data['lag_dist'].astype('int32')

    # gw_depths = gw_data[['ref', 'depth']].copy().load()
    # depths = gw_depths.depth.values

    # for ref in gw_data.ref.values:

    gw_data['depth_cat'] = gw_data.depth.astype('int16')
    gw_data = gw_data.assign({'depth_cat': (('ref'), np.array([find_nearest(lag_depths, v) for v in gw_data.depth.values], dtype='int16'))})

    gw_data1 = gw_data.sel(lag_depth=gw_data['depth_cat'], drop=True).copy().load()

    lag_dist = []
    lag_mins = []
    lag_maxes = []
    lag_medians = []
    depth_mins = []
    depth_maxes = []
    for ref in gw_data1.ref.values:
        data = gw_data1.sel(ref=ref, drop=True).dropna('lag_dist').isel(lag_dist=0)
        lag_mins.append(int(data.lag_min.values))
        lag_maxes.append(int(data.lag_max.values))
        lag_medians.append(int(data.lag_median.values))
        lag_dist.append(int(data.lag_dist.values))
        depth = int(data.depth.values)
        depth_cat = int(data.depth_cat.values)
        depth_min = depth - depth_cat
        if depth_min < 0:
            depth_min = 0
        depth_max = depth + depth_cat

        depth_mins.append(depth_min)
        depth_maxes.append(depth_max)

    gw_data2 = gw_data1.drop(['lag_dist', 'lag_depth']).assign({'lag_min': (('ref'), lag_mins), 'lag_max': (('ref'), lag_maxes), 'lag_median': (('ref'), lag_medians), 'lag_dist': (('ref'), lag_dist), 'depth_min': (('ref'), depth_mins), 'depth_max': (('ref'), depth_maxes)})

    ## Make the geometry
    gw_pts0 = gw_data2[['lon', 'lat', 'depth', 'lag_at_site', 'lag_min', 'lag_max', 'lag_median', 'lag_dist', 'depth_min', 'depth_max']].to_dataframe().reset_index()
    gw_data.close()

    gw_pts1 = vector.xy_to_gpd(['ref', 'depth', 'lag_at_site', 'lag_min', 'lag_max', 'lag_median', 'lag_dist', 'depth_min', 'depth_max'], 'lon', 'lat', gw_pts0, 4326)
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
    gw_pts1.drop('tooltip', axis=1).rename(columns={'ref': 'well_id', 'depth': 'well_depth', 'lag_at_site': 'MRT_at_well', 'lag_min': 'external_MRT_min', 'lag_max': 'external_MRT_max', 'lag_median': 'external_MRT_median', 'lag_dist': 'external_MRT_distance'}).to_file(utils.gw_points_path)

    with booklet.open(utils.gw_points_rc_blt, 'n', value_serializer=None, key_serializer='str', n_buckets=101) as gw:
        for name, gbuf in gw_dict.items():
            gw[name] = gbuf

    rc3 = rc1.copy()
    rc3['geometry'] = rc1.buffer(0).simplify(30)

    rc_geo = rc3.to_crs(4326).set_index('name', drop=False).__geo_interface__
    rc_gbuf = geobuf.encode(rc_geo)

    with open(utils.rc_bounds_gbuf, 'wb') as f:
        f.write(rc_gbuf)






















































