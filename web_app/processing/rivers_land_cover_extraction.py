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
import utils
import shelflet
import shelve

pd.options.display.max_columns = 10

##########################################
### parcels

# parcels = gpd.read_file(utils.parcels_path)
# parcels = parcels[['id', 'geometry']].copy()

## Separate parcels and land cover into catchments
catch0 = utils.read_pkl_zstd(utils.output_path.joinpath(utils.major_catch_file), True)

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

with shelflet.open(utils.catch_lc_path) as land_cover_dict:
    for i, row in catch0.iterrows():
        print(i)

        # Land cover
        lc2 = land_cover.loc[land_cover.sindex.query(row.geometry, predicate="intersects")].copy()
        lc2b = intersection(lc2.geometry.tolist(), row.geometry)
        lc2['geometry'] = lc2b
        land_cover_dict[str(row.nzsegment)] = lc2.simplify(30)
        land_cover_dict.sync()

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









































































































