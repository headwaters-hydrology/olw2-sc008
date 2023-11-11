#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:47:16 2023

@author: mike
"""

import sys
import os
import pandas as pd
import pathlib
import geopandas as gpd
import booklet
import geobuf
from gistools import vector
import orjson

if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

######################################################
### Parameters

#####################################################
### Process data


def rivers_marae_processing():
    """

    """
    marae0 = gpd.read_file(utils.marae_gpkg)
    marae0['geometry'] = marae0.simplify(1)
    marae1 = marae0[['GIS_MID', 'Name', 'geometry']].rename(columns={'Name': 'tooltip'}).set_index('GIS_MID').to_crs(4326).copy()

    ## Organise by catchment
    catches0 = booklet.open(utils.river_catch_major_path)

    with booklet.open(utils.river_marae_path, 'n', key_serializer='uint4', value_serializer='zstd', n_buckets=1607) as f:
        for k, v in catches0.items():
            marae2 = marae1[marae1.within(v)]
            marae3 = orjson.loads(marae2.to_json())
            gbuf = geobuf.encode(marae3)
            f[k] = gbuf
            # f[k] = marae3

    catches0.close()











































