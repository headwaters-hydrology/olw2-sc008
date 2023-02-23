#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 08:57:22 2022

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
# import dbm
import utils
import booklet
# import shelve
import multiprocessing as mp
import concurrent.futures
import pickle

pd.options.display.max_columns = 10

#############################################
### Combine land cover and parcels


# def overlay_parcels_lc(row, catch_parcels_path, catch_lc_path, red1, lc_red_dict):
#     """

#     """
#     seg = str(row.nzsegment)
#     print(seg)

#     parcels_dict = booklet.open(utils.catch_parcels_path, 'r')
#     land_cover_dict = booklet.open(utils.catch_lc_path, 'r')

#     parcels = parcels_dict[seg]

#     lc = land_cover_dict[seg].copy()
#     lc.rename(columns={'Name_2018': 'land_cover'}, inplace=True)
#     lc_names = lc['land_cover'].tolist()

#     new_names = []
#     for name in lc_names:
#         if name not in lc_red_dict:
#             new_name = 'Other'
#         else:
#             new_name = name
#         new_names.append(new_name)

#     lc['land_cover'] = new_names

#     lc2 = lc.dissolve('land_cover').reset_index()

#     combo0 = parcels.overlay(lc2)
#     combo1 = combo0.merge(red1.reset_index(), on='land_cover')

#     parcels_dict.close()
#     land_cover_dict.close()

#     return seg, combo1


def land_cover_process(catch_id, lc, red1, lc_red_dict):
    """

    """
    # print(catch_id)

    lc.rename(columns={'Name_2018': 'land_cover'}, inplace=True)
    lc_names = lc['land_cover'].tolist()

    new_names = []
    for name in lc_names:
        if name not in lc_red_dict:
            new_name = 'Other'
        else:
            new_name = name
        new_names.append(new_name)

    lc['land_cover'] = new_names

    lc['geometry'] = lc['geometry'].buffer(0)

    lc2 = lc.dissolve('land_cover').reset_index()
    lc2['geometry'] = lc2.simplify(20)
    combo1 = lc2.merge(red1.reset_index(), on='land_cover')

    return catch_id, combo1



if __name__ == '__main__':
    # catch0 = gpd.read_feather(utils.major_catch_file)
    # catch0['geometry'] = catch0['geometry'].buffer(0)

    # catch0 = booklet.open(utils.river_catch_major_path)

    land_cover = booklet.open(utils.catch_lc_path, 'r')

    lc_red_dict = utils.land_cover_reductions.copy()
    red1 = pd.DataFrame.from_dict(lc_red_dict, orient='index', columns=['reduction'])
    red1.index.name = 'land_cover'

    # combo_list = []
    # for i, row in catch0.iterrows():
    #     output = land_cover_process(row, red1, lc_red_dict)
    #     combo_list.append(output)


    with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for catch_id, lc in land_cover.items():
            f = executor.submit(land_cover_process, catch_id, lc, red1, lc_red_dict)
            futures.append(f)

        runs = concurrent.futures.wait(futures)

    combo_list = [r.result() for r in runs[0]]

    ## Save data
    with booklet.open(utils.catch_lc_clean_path, 'n', value_serializer='gpd_zstd', key_serializer='uint4') as parcels_lc:
        for seg, data in combo_list:
            parcels_lc[seg] = data






#########################################3
### Testing

# seg = '14330003'

# row = catch0[catch0.nzsegment == int(seg)].iloc[0]

# segs = set(catch0.nzsegment.astype(str).tolist())

# parcels_lc = booklet.open(utils.catch_parcels_lc_path, 'r')

# p_segs = set(parcels_lc.keys())

# diff = segs.difference(p_segs)


# p1 = parcels.geometry.tolist()

# new1 = intersection(p1, lc2.geometry.tolist())

# lc = booklet.open(utils.catch_lc_clean_path, 'r')

# n1 = lc[seg]

# lc.close()













































