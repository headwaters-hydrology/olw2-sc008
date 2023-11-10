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

#####################################################
### Process data


def eco_monitoring_sites_processing():
    data_list = []
    for param, val in utils.eco_moni_data_dict.items():
        sites0 = pd.read_csv(utils.eco_data_path.joinpath(val['file_name']))
        cols = [col for col in val.values() if col in sites0]
        sites1 = sites0[cols].dropna().rename(columns={v: k for k, v in val.items()}).copy()
        sites1 = sites1[sites1.stdev > 0].copy()
        sites1['indicator'] = param
        data_list.append(sites1)

    sites2 = pd.concat(data_list)
    sites2 = sites2[sites2.n_sites > 9].copy()

    ### Site locations
    sites3 = sites2.drop_duplicates(subset=['nzsegment'], keep='first').drop(['stdev', 'mean', 'indicator'], axis=1)
    sites3.loc[sites3.site_id.isnull(), 'site_id'] = 'Unknown'

    sites4 = vector.xy_to_gpd(['site_id', 'nzsegment'], 'nztmx', 'nztmy', sites3, 2193).to_crs(4326)
    sites4['nzsegment'] = sites4['nzsegment'].astype('int32')

    sites4.to_file(utils.eco_sites_gpkg_path)

    ## Organise by catchment
    catches0 = booklet.open(utils.river_catch_major_path)

    sites2['catch_id'] = 0
    with booklet.open(utils.eco_sites_catch_path, 'n', key_serializer='uint4', value_serializer='zstd', n_buckets=1600) as f:
        for k, v in catches0.items():
            sites5 = sites4[sites4.within(v)]
            sites2.loc[sites2.nzsegment.isin(sites5.nzsegment), 'catch_id'] = k
            sites6 = sites5.set_index('site_id', drop=False).rename(columns={'site_id': 'tooltip'}).__geo_interface__
            gbuf = geobuf.encode(sites6)
            f[k] = gbuf

    catches0.close()

    ### Sites stdev processing
    stdev0 = sites2[sites2.catch_id > 0].drop(['site_id', 'nztmx', 'nztmy'], axis=1).copy()
    stdev0['nzsegment'] = stdev0['nzsegment'].astype('int32')
    stdev0['stdev'] = stdev0['stdev']/stdev0['mean']
    stdev1 = stdev0.drop('mean', axis=1)
    stdev1.to_csv(utils.eco_moni_stdev_path, index=False)












