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
import nzrec

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

corrections = {'Motupiko at 250m u-s Motueka Rv': 10021971}

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

    for site_id, seg in corrections.items():
        sites1.loc[sites1.site_id == site_id, 'nzsegment'] = seg

    ## Assign stream orders
    w0 = nzrec.Water(utils.nzrec_data_path)

    segs = []
    append = segs.append
    for seg in sites1.nzsegment:
        append(w0._way_tag[seg]['Strahler stream order'])

    sites1['stream_order'] = segs

    sites2 = vector.xy_to_gpd(['site_id', 'lawa_id', 'nzsegment', 'stream_order'], 'lon', 'lat', sites1, 4326)

    sites2.to_file(utils.river_sites_path)

    ## Organise by catchment
    catches0 = booklet.open(utils.river_catch_major_path)

    catches_all = {}
    catches_3rd = {}
    for k, v in catches0.items():
        sites3 = sites2[sites2.within(v)]
        sites4 = sites3.set_index('site_id', drop=False).rename(columns={'site_id': 'tooltip'})
        catches_all[k] = sites4
        catches_3rd[k] = sites4[sites4.stream_order >= 3].copy()

    catches0.close()

    with booklet.open(utils.river_sites_catch_path, 'n', key_serializer='uint4', value_serializer='zstd', n_buckets=1607) as f:
        for k, v in catches_all.items():
            v2 = v.__geo_interface__
            gbuf = geobuf.encode(v2)
            f[k] = gbuf

    with booklet.open(utils.river_sites_catch_3rd_path, 'n', key_serializer='uint4', value_serializer='zstd', n_buckets=1607) as f:
        for k, v in catches_3rd.items():
            v2 = v.__geo_interface__
            gbuf = geobuf.encode(v2)
            f[k] = gbuf







