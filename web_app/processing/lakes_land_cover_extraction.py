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
import booklet
import pickle
import zstandard as zstd

import utils


pd.options.display.max_columns = 10


##########################################
### parcels

# parcels = gpd.read_file(utils.parcels_path)
# parcels = parcels[['id', 'geometry']].copy()

## Separate parcels and land cover into catchments
# catch0 = utils.read_pkl_zstd(utils.output_path.joinpath(utils.major_catch_file), True)

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

def lakes_lc_process():
    lc_red_dict = utils.land_cover_reductions.copy()
    red1 = pd.DataFrame.from_dict(lc_red_dict, orient='index', columns=['reduction'])
    red1.index.name = 'land_cover'

    ## land cover
    land_cover = gpd.read_file(utils.land_cover_path)
    land_cover = land_cover[['Name_2018', 'geometry']].rename(columns={'Name_2018': 'land_cover'}).copy()

    with booklet.open(utils.lakes_lc_path, 'n', value_serializer='gpd_zstd', key_serializer='uint2', n_buckets=400) as land_cover_dict:
        with booklet.open(utils.lakes_catches_minor_path) as f:
            for lake in f:
                print(lake)

                catch = f[lake]

                # Land cover
                lc2 = land_cover.loc[land_cover.sindex.query(catch.unary_union, predicate="intersects")].copy()
                lc2b = intersection(lc2.geometry.tolist(), catch.unary_union)
                lc2['geometry'] = lc2b

                lc_names = lc2['land_cover'].tolist()

                new_names = []
                for name in lc_names:
                    if name not in lc_red_dict:
                        new_name = 'Other'
                    else:
                        new_name = name
                    new_names.append(new_name)

                lc2['land_cover'] = new_names
                lc2['geometry'] = lc2.buffer(0)

                lc3 = lc2.dissolve('land_cover').reset_index()
                lc3['geometry'] = lc3.simplify(20)
                combo1 = lc3.merge(red1.reset_index(), on='land_cover')

                land_cover_dict[lake] = combo1

    # close/delete objects
    del land_cover


##############################################
### Testing

# land_cover_dict = booklet.open(utils.lakes_lc_path, 'r')


# with shelve.open('/media/nvme1/data/OLW/web_app/output/shelve_test.shelf') as t:
#     for seg in land_cover_dict:
#         lc2 = land_cover_dict[seg]
#         t[seg] = lc2
#         t.sync()

# file = '/media/nvme1/data/OLW/web_app/output/assets/lakes_lc.blt'

# land_cover_dict = open(file, 'n', value_serializer=PickleZstd, key_serializer=Str, n_buckets=200)


# f = booklet.open(utils.lakes_catches_minor_path)



































































































