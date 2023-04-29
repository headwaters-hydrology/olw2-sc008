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
### land use/cover

# parcels = gpd.read_file(utils.parcels_path)
# parcels = parcels[['id', 'geometry']].copy()

lcdb_classes = ['Exotic Forest', 'Forest - Harvested', 'Orchard, Vineyard or Other Perennial Crop', 'Short-rotation Cropland']

lcdb_where_sql = 'Name_2018 IN ({})'.format(str(lcdb_classes)[1:-1])


def land_cover_processing():
    ## land cover
    lcdb0 = gpd.read_file(utils.land_cover_path, include_fields=['Name_2018'], where=lcdb_where_sql)
    lcdb0['geometry'] = lcdb0.buffer(0).simplify(1)
    lcdb0 = lcdb0.rename(columns={'Name_2018': 'land_cover'})
    lcdb0['farm_type'] = 'NA'
    lcdb0['typology'] = lcdb0['land_cover']

    ## SnB and dairy
    snb0 = gpd.read_file(utils.snb_geo_path, include_fields=['FARM_TYPE', 'Typology'])
    snb0['geometry'] = snb0.buffer(0).simplify(1)
    snb0 = snb0.rename(columns={'FARM_TYPE': 'farm_type', 'Typology': 'typology'})
    snb0['land_cover'] = 'Sheep and Beef'
    snb0.loc[snb0['farm_type'].isnull(), 'farm_type'] = 'NA'

    snb0['typology'] = snb0.typology.apply(lambda x: x.split('.I. ')[1])

    utils.gpd_to_feather(snb0, utils.snb_geo_clean_path)

    dairy0 = gpd.read_file(utils.dairy_geo_path, include_fields=['Farmtype_3', 'Typology'])
    dairy0['geometry'] = dairy0.buffer(0).simplify(1)
    dairy0 = dairy0.rename(columns={'Farmtype_3': 'farm_type', 'Typology': 'typology'})
    dairy0['typology'] = dairy0['typology'].str.replace('Steep', 'High')
    dairy0['land_cover'] = 'Dairy'
    dairy0.loc[dairy0['farm_type'].isnull(), 'farm_type'] = 'NA'

    utils.gpd_to_feather(dairy0, utils.dairy_geo_clean_path)

    combo1 = pd.concat([snb0, dairy0])

    ## Symmetric difference to the lcdb
    lcdb1 = lcdb0.overlay(combo1, how='difference')

    ## combine all and dissolve
    combo2 = pd.concat([combo1, lcdb1])
    # combo3 = combo2.dissolve('typology')

    ## Save
    # utils.gpd_to_feather(combo3.reset_index(), utils.lc_clean_path)
    utils.gpd_to_feather(combo2, utils.lc_clean_diss_path)
    # combo2.to_file(utils.lc_clean_gpkg_path)


































































































