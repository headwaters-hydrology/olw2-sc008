#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:52:39 2023

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

if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

######################################################
### Parameters

test_sites = {
    'Rai River at Rai Falls': 11015032,
    'Opouri River at Tunakino Valley Road': 11012404,
    'Ronga River at Upstream Rai River': 11012146
    }



#####################################################
### Process data


def rivers_monitoring_sites_processing():
    sites0a = pd.read_csv(utils.sites_names_csv)
    sites0a['lawa_id'] = sites0a['lawa_id'].str.upper()
    sites0b = pd.read_csv(utils.sites_rec_csv).dropna()
    sites0b['lawa_id'] = sites0b['lawa_id'].str.upper()
    sites0b['nzsegment'] = sites0b['nzsegment'].astype('int32')
    sites1 = pd.merge(sites0b, sites0a, on='lawa_id', how='left')
    sites1.loc[sites1.site_id.isnull(), 'site_id'] = sites1.loc[sites1.site_id.isnull(), 'lawa_id']

    sites2 = vector.xy_to_gpd(['site_id', 'lawa_id', 'nzsegment'], 'lon', 'lat', sites1, 4326)

    sites2.to_file(utils.river_sites_path)

    ## Organise by catchment
    catches0 = booklet.open(utils.river_catch_major_path)
    # sites1 = sites0.to_crs(4326)

    with booklet.open(utils.river_sites_catch_path, 'n', key_serializer='uint4', value_serializer='zstd', n_buckets=1607) as f:
        for k, v in catches0.items():
            sites3 = sites2[sites2.within(v)]
            sites4 = sites3.set_index('site_id', drop=False).rename(columns={'site_id': 'tooltip'}).__geo_interface__
            gbuf = geobuf.encode(sites4)
            f[k] = gbuf

    catches0.close()





