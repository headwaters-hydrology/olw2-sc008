#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:44:19 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
import booklet
import geobuf
import pickle
import zstandard as zstd
import base64

import utils

pd.options.display.max_columns = 10


#############################################
### Lakes

def lakes_catch_delin():
    rec_rivers0 = gpd.read_feather(utils.rec_rivers_feather)
    rec_catch0 = gpd.read_feather(utils.rec_catch_feather)

    delin_points = gpd.read_file(utils.lakes_delin_points_path)

    catches_major_dict = {}
    catches_minor_dict = {}
    segs_dict = {}
    reaches_dict = {}
    for name, points in delin_points.groupby('name'):
        print(name)
        catches, reaches, pts_seg = rec.catch_delineate(points, rec_rivers0, rec_catch0, segment_id_col='nzsegment', from_node_col='from_node', to_node_col='to_node', ignore_order=1, stream_order_col='stream_order', max_distance=1000, site_delineate='all', returns='all')

        catch = catches.iloc[[0]][['name', 'geometry']].set_index('name').copy()
        catch['geometry'] = catches.unary_union

        gbuf = geobuf.encode(catch.simplify(30).to_crs(4326).__geo_interface__)

        segs = reaches.nzsegment.unique().astype('int32')

        catches_major_dict[name] = gbuf
        segs_dict[name] = segs

        minor_catch1 = rec_catch0[rec_catch0.nzsegment.isin(segs)][['nzsegment', 'geometry']].copy()
        minor_catch1['geometry'] = minor_catch1.simplify(0)
        catches_minor_dict[name] = minor_catch1

        reaches1 = rec_rivers0[rec_rivers0.nzsegment.isin(segs)].copy()[['nzsegment', 'stream_order', 'geometry']].set_index('nzsegment', drop=False)
        reaches1 = reaches1[reaches1.stream_order > 2].copy()
        reaches1['geometry'] = reaches1.simplify(30)
        reaches1 = reaches1.to_crs(4326)

        gbuf = geobuf.encode(reaches1.__geo_interface__)

        reaches_dict[name] = gbuf


    ## Save results
    with booklet.open(utils.lakes_reaches_mapping_path, 'n', value_serializer='numpy_int4_zstd', key_serializer='str', n_buckets=200) as mapping:
        for name in segs_dict:
            mapping[name] = segs_dict[name]

    with booklet.open(utils.lakes_catches_major_path, 'n', value_serializer='zstd', key_serializer='str', n_buckets=200) as mapping:
        for name in catches_major_dict:
            mapping[name] = catches_major_dict[name]

    with booklet.open(utils.lakes_catches_minor_path, 'n', value_serializer='gpd_zstd', key_serializer='str', n_bytes_key=1, n_buckets=200) as mapping:
        for name in catches_minor_dict:
            mapping[name] = catches_minor_dict[name]

    with booklet.open(utils.lakes_reaches_path, 'n', value_serializer='zstd', key_serializer='str', n_buckets=200) as mapping:
        for name in reaches_dict:
            mapping[name] = reaches_dict[name]












































































