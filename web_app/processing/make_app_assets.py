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
import utils

pd.options.display.max_columns = 10

##############################################
### Functions

def process_assets():
    """

    """
    reaches2 = utils.read_pkl_zstd(utils.output_path.joinpath(utils.rec_delin_file), True)

    ## Save reaches geobuf
    reaches3 = reaches2.loc[reaches2.stream_order > 2].to_crs(4326)
    starts = reaches3.start.unique()

    for s in starts:
        df1 = reaches3[reaches3['start'] == s].drop('start', axis=1).set_index('nzsegment', drop=False)
        gjson = orjson.loads(df1.to_json())
        gbuf = geobuf.encode(gjson)

        new_path = utils.reach_gbuf_path.joinpath(str(s) + '.pbf')
        with open(new_path, 'wb') as f:
            f.write(gbuf)


    ## Save catchments geobufs
    rec_shed = utils.read_pkl_zstd(utils.output_path.joinpath(utils.major_catch_file), True)

    gjson = orjson.loads(rec_shed.set_index('nzsegment').to_crs(4326).to_json())

    with open(utils.assets_path.joinpath('catchments.pbf'), 'wb') as f:
        f.write(geobuf.encode(gjson))


    rec_catch2 = utils.read_pkl_zstd(utils.output_path.joinpath(utils.catch_file), True)

    grp1 = rec_catch2.groupby('start')

    for catch_id, catches in grp1:
        new_path = utils.catch_path.joinpath(str(catch_id) + '.pkl.zst')
        utils.write_pkl_zstd(catches, new_path)

    ## Save reach mappings
    mapping = utils.read_pkl_zstd(utils.output_path.joinpath(utils.reach_mapping_file), True)

    for catch_id, reaches in mapping.items():
        new_path = utils.reach_map_path.joinpath(str(catch_id) + '.pkl.zst')
        utils.write_pkl_zstd(reaches, new_path)




























