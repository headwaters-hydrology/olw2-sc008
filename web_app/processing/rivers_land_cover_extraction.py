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

import utils

pd.options.display.max_columns = 10

##########################################
### parcels

# parcels = gpd.read_file(utils.parcels_path)
# parcels = parcels[['id', 'geometry']].copy()

def rivers_land_cover():
    ## Separate parcels and land cover into catchments
    catch0 = gpd.read_feather(utils.major_catch_file)
    catch0['geometry'] = catch0.buffer(0)

    # with shelflet.open(utils.catch_parcels_path) as parcels_dict:
    #     for i, row in catch0.iterrows():
    #         print(i)

    #         # Parcels
    #         parcels2 = parcels.loc[parcels.sindex.query(row.geometry, predicate="intersects")].copy()
    #         parcels2b = intersection(parcels2.geometry.tolist(), row.geometry)
    #         parcels2['geometry'] = parcels2b
    #         parcels_dict[str(row.nzsegment)] = parcels2
    #         parcels_dict.sync()

    # # close/delete objects
    # del parcels

    ## land cover
    land_cover = gpd.read_file(utils.land_cover_path)
    land_cover = land_cover[['Name_2018', 'geometry']].copy()
    land_cover['geometry'] = land_cover.buffer(0)

    lc_dict = {}
    for i, row in catch0.iterrows():
        print(i)

        # Land cover
        lc2 = land_cover.loc[land_cover.sindex.query(row.geometry, predicate="intersects")].copy()
        # lc2['geometry'] = lc2['geometry'].simplify(0)
        lc2b = intersection(lc2.geometry.tolist(), row.geometry)
        lc2['geometry'] = lc2b
        lc2['geometry'] = lc2['geometry'].simplify(30)
        lc_dict[row.nzsegment] = lc2

    print('save file')
    with booklet.open(utils.catch_lc_path, 'n', value_serializer='gpd_zstd', key_serializer='uint4', n_buckets=2000) as land_cover_dict:
        for i, lc2 in lc_dict.items():
            land_cover_dict[i] = lc2

    # close/delete objects
    del land_cover


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

































































































