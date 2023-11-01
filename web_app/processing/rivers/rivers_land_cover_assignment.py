#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:11:07 2022

@author: mike
"""
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import intersection, difference, intersects
import booklet
import orjson
import geobuf

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

##########################################
### parcels

# parcels = gpd.read_file(utils.parcels_path)
# parcels = parcels[['id', 'geometry']].copy()

# way_id = 3076139
# catch = catches[way_id]

line_break = '<br />'
bold_start = '<b>'
bold_end = '</b>'


def make_tooltip(x):
    """

    """
    tt = '<b>Typology</b><br />{typo}<br /><b>Land Cover</b><br />{lc}'.format(typo=x['typology'], lc=x['land_cover'])

    return tt


def rivers_land_cover():
    ## Separate land cover into catchments
    catches = booklet.open(utils.river_catch_major_path)

    ## land cover
    # land_cover = gpd.read_feather(utils.lc_red_feather_path)
    lcdb0 = gpd.read_feather(utils.lcdb_red_path)
    snb_dairy0 = gpd.read_feather(utils.snb_dairy_red_path)

    lc_dict = {}
    for way_id, catch in catches.items():
        print(way_id)

        c1 = gpd.GeoSeries([catch], crs=4326).to_crs(2193).iloc[0]

        # Land cover
        lcdb1 = lcdb0.loc[lcdb0.sindex.query(c1, predicate="intersects")].copy()
        if not lcdb1.empty:
            lcdb1b = intersection(lcdb1.geometry.tolist(), c1)
            lcdb1['geometry'] = lcdb1b
            lcdb1 = lcdb1[~lcdb1.geometry.is_empty].copy()
            lcdb1 = lcdb1.dissolve('typology').reset_index()
            lcdb1['geometry'] = lcdb1.buffer(0.1).simplify(10).make_valid()

            # lc3['geometry'] = lc3.buffer(0.5).simplify(10)

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

    print('save files')
    with booklet.open(utils.catch_lc_path, 'n', value_serializer='gpd_zstd', key_serializer='uint4', n_buckets=1601) as land_cover_dict:
        for i, lc2 in lc_dict.items():
            land_cover_dict[i] = lc2

    with booklet.open(utils.catch_lc_pbf_path, 'n', value_serializer=None, key_serializer='uint4', n_buckets=1601) as lc_gbuf:
        for i, lc2 in lc_dict.items():
            lc2['tooltip'] = lc2.apply(lambda x: make_tooltip(x), axis=1)
            gdf = lc2.to_crs(4326)
            gjson = orjson.loads(gdf.to_json())
            gbuf = geobuf.encode(gjson)
            lc_gbuf[i] = gbuf

    for i, data in lc_dict.items():
        path = utils.rivers_catch_lc_dir.joinpath(utils.rivers_catch_lc_gpkg_str.format(i))
        data.drop('tooltip', axis=1).to_file(path)

    combo_list = []
    with booklet.open(utils.catch_lc_path) as lc:
        for i, data in lc.items():
            if not data.empty:
                data.insert(0, 'catch_id', i)
                combo_list.append(data)

    combo1 = pd.concat(combo_list)
    combo1.to_file(utils.rivers_catch_lc_gpkg_path)


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


# with booklet.open(utils.catch_lc_path) as lc_dict:
#     with booklet.open(utils.catch_lc_pbf_path, 'n', value_serializer=None, key_serializer='uint4', n_buckets=1600) as lc_gbuf:
#         for i, lc2 in lc_dict.items():
#             gdf = lc2.to_crs(4326)
#             gjson = orjson.loads(gdf.to_json())
#             gbuf = geobuf.encode(gjson)
#             lc_gbuf[i] = gbuf






























































































