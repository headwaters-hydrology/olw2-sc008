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
import shelflet
# import shelve
import multiprocessing as mp
import concurrent.futures

pd.options.display.max_columns = 10

#############################################
### Combine land cover and parcels


def overlay_parcels_lc(row, catch_parcels_path, catch_lc_path, red1, lc_red_dict):
    """

    """
    seg = str(row.nzsegment)
    print(seg)

    parcels_dict = shelflet.open(utils.catch_parcels_path, 'r')
    land_cover_dict = shelflet.open(utils.catch_lc_path, 'r')

    parcels = parcels_dict[seg]

    lc = land_cover_dict[seg].copy()
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

    lc2 = lc.dissolve('land_cover').reset_index()

    combo0 = parcels.overlay(lc2)
    combo1 = combo0.merge(red1.reset_index(), on='land_cover')

    parcels_dict.close()
    land_cover_dict.close()

    return seg, combo1


if __name__ == '__main__':
    catch0 = utils.read_pkl_zstd(utils.output_path.joinpath(utils.major_catch_file), True)

    lc_red_dict = utils.land_cover_reductions.copy()
    red1 = pd.DataFrame.from_dict(lc_red_dict, orient='index', columns=['reduction'])
    red1.index.name = 'land_cover'

    with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for i, row in catch0.iterrows():
            f = executor.submit(overlay_parcels_lc, row, utils.catch_parcels_path, utils.catch_lc_path, red1, lc_red_dict)
            futures.append(f)

        runs = concurrent.futures.wait(futures)

    combo_list = [r.result() for r in runs[0]]


    ## Save data
    with shelflet.open(utils.catch_parcels_lc_path) as parcels_lc:
        for seg, data in combo_list:
            parcels_lc[seg] = data
            parcels_lc.sync()





































































