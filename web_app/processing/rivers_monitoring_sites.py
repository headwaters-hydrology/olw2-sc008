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
import booklet
import geobuf

import utils

pd.options.display.max_columns = 10

######################################################
### Parameters

#####################################################
### Process data


def rivers_monitoring_sites_processing():
    catches0 = booklet.open(utils.river_catch_major_path)

    sites0 = gpd.read_feather(utils.river_sites_path)
    sites1 = sites0.to_crs(4326)

    with booklet.open(utils.river_sites_catch_path, 'n', key_serializer='uint4', value_serializer='zstd', n_buckets=1600) as f:
        for k, v in catches0.items():
            sites2 = sites1[sites1.within(v)]
            sites3 = sites2.set_index('site_id', drop=False).rename(columns={'site_id': 'tooltip'}).__geo_interface__
            gbuf = geobuf.encode(sites3)
            f[k] = gbuf

    catches0.close()





