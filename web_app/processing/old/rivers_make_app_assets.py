#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:16:40 2022

@author: mike
"""
import pathlib
import os
import pickle
import io
from shapely.ops import unary_union
import geobuf
import base64
import orjson
import zstandard as zstd
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
import booklet

import utils

pd.options.display.max_columns = 10

##############################################
### Functions


def process_assets():
    """

    """
    reaches2 = gpd.read_feather(utils.rec_delin_file)

    ## Save reaches geobuf
    reaches3 = reaches2.loc[reaches2.stream_order > 2].to_crs(4326)
    starts = reaches3.start.unique()

    with booklet.open(utils.river_reach_gbuf_path, 'n', key_serializer='uint4', value_serializer='zstd') as f:
        for s in starts:
            df1 = reaches3[reaches3['start'] == s].drop('start', axis=1).set_index('nzsegment', drop=False)
            gjson = orjson.loads(df1.to_json())
            gbuf = geobuf.encode(gjson)

            f[int(s)] = gbuf

    ## Save catchments geobufs
    rec_shed = gpd.read_feather(utils.major_catch_file)

    gjson = orjson.loads(rec_shed.set_index('nzsegment').to_crs(4326).to_json())

    with open(utils.assets_path.joinpath('catchments.pbf'), 'wb') as f:
        f.write(geobuf.encode(gjson))


    rec_catch2 = gpd.read_feather(utils.catch_file)

    grp1 = rec_catch2.groupby('start')

    with booklet.open(utils.river_catch_path, 'n', key_serializer='uint4', value_serializer='gpd_zstd') as f:
        for catch_id, catches in grp1:
            catches1 = catches.drop(['start', 'stream_order'], axis=1)
            catches1['geometry'] = catches1.simplify(20)
            f[catch_id] = catches1

    ## Save reach mappings
    # mapping = gpd.read_feather(utils.reach_mapping_file)

    # with booklet.open(utils.river_reach_mapping_path, 'n') as f:
    #     for catch_id, reaches in mapping.items():
    #         f[catch_id] = reaches




# f = booklet.open(utils.river_catch_path)

# keys = list(f.keys())

# for key in keys:
#     r1 = f[key]



















