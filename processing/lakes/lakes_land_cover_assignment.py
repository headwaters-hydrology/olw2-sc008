#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:11:07 2022

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
import dbm
import booklet
import pickle
import zstandard as zstd

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

##########################################
### parcels

# parcels = gpd.read_file(utils.parcels_path)
# parcels = parcels[['id', 'geometry']].copy()

def lakes_land_cover():
    ## Separate land cover into catchments
    catches = booklet.open(utils.lakes_catches_minor_path)

    ## land cover
    lcdb0 = gpd.read_feather(utils.lcdb_red_path)
    snb_dairy0 = gpd.read_feather(utils.snb_dairy_red_path)

    lc_dict = {}
    for way_id, catch in catches.items():
        print(way_id)

        c1 = catch.unary_union

        # Land cover
        lcdb1 = lcdb0.loc[lcdb0.sindex.query(c1, predicate="intersects")].copy()
        if not lcdb1.empty:
            lcdb1b = intersection(lcdb1.geometry.tolist(), c1)
            lcdb1['geometry'] = lcdb1b
            lcdb1 = lcdb1[~lcdb1.geometry.is_empty].copy()
            lcdb1 = lcdb1.dissolve('typology').reset_index()
            lcdb1['geometry'] = lcdb1.buffer(0.1).simplify(10).make_valid()

        ## SnB and Dairy
        snb_dairy1 = snb_dairy0.loc[snb_dairy0.sindex.query(c1, predicate="intersects")].copy()
        if not snb_dairy1.empty:
            snb_dairy1b = intersection(snb_dairy1.geometry.tolist(), c1)
            snb_dairy1['geometry'] = snb_dairy1b
            snb_dairy1 = snb_dairy1[~snb_dairy1.geometry.is_empty].copy()
            snb_dairy1 = snb_dairy1.dissolve('typology').reset_index()
            snb_dairy1['geometry'] = snb_dairy1['geometry'].buffer(0.1).simplify(10).make_valid()

        if (not snb_dairy1.empty) and (not lcdb1.empty):
            # diff_list = []
            # for geo1 in lcdb1.geometry:
            #     for geo2 in snb_dairy1.geometry:
            #         if intersects(geo1, geo2):
            #             geo3 = difference(geo1, geo2)
            #             diff_list.append(geo3)
            #         else:
            #             diff_list.append(geo1)
            lcdb1 = lcdb1.overlay(snb_dairy1, how='difference', keep_geom_type=True)
            combo2 = pd.concat([snb_dairy1, lcdb1])
        elif snb_dairy1.empty:
            combo2 = lcdb1
        else:
            combo2 = snb_dairy1

        # lc2['geometry'] = lc2['geometry'].simplify(30)
        lc_dict[way_id] = combo2

    catches.close()

    print('save file')
    with booklet.open(utils.lakes_lc_path, 'n', value_serializer='gpd_zstd', key_serializer='uint4', n_buckets=401) as land_cover_dict:
        for i, lc2 in lc_dict.items():
            land_cover_dict[i] = lc2

    with booklet.open(utils.lakes_lc_path) as lc:
        for i, data in lc.items():
            path = utils.lakes_catch_lc_dir.joinpath(utils.lakes_catch_lc_gpkg_str.format(i))
            data.to_file(path)



##############################################
### Testing

# land_cover_dict = shelflet.open(utils.catch_lc_path, 'r')


# with shelve.open('/media/nvme1/data/OLW/web_app/output/shelve_test.shelf') as t:
#     for seg in land_cover_dict:
#         lc2 = land_cover_dict[seg]
#         t[seg] = lc2
#         t.sync()



# db = booklet.open(utils.catch_lc_path)

# keys = list(db.keys())

# db[9259625]

































































































