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

lcdb_reductions = {'Exotic Forest': 0, 'Forest - Harvested': 0, 'Orchard, Vineyard or Other Perennial Crop': 10, 'Short-rotation Cropland': 30}


def land_cover_reductions():
    ## land cover
    lc0 = gpd.read_feather(utils.lc_clean_path)

    ## Apply the default reductions
    # SnB
    snb1 = pd.read_csv(utils.snb_typo_path, header=[0, 1])
    phos1 = snb1['Phosphorus'].copy()
    phos2 = ((1 - (phos1['2035 potential load (kg)'] / phos1['2015 current load (kg)'])) * 100).round().astype('int8')
    phos2.name = 'phosphorus'
    nitrate1 = snb1['Nitrogen'].copy()
    nitrate2 = ((1 - (nitrate1['2035 potential load (kg)'] / nitrate1['2015 current load (kg)'])) * 100).round().astype('int8')
    nitrate2.name = 'nitrogen'
    snb2 = pd.concat([snb1['typology']['typology'], phos2, nitrate2], axis=1)

    # dairy
    dairy1 = pd.read_csv(utils.dairy_typo_path, header=[0, 1])
    phos1 = dairy1['Phosphorus'].copy()
    phos2 = ((1 - (phos1['2035 potential load (kg)'] / phos1['2015 current load (kg)'])) * 100).round().astype('int8')
    phos2.name = 'phosphorus'
    nitrate1 = dairy1['Nitrogen'].copy()
    nitrate2 = ((1 - (nitrate1['2035 potential load (kg)'] / nitrate1['2015 current load (kg)'])) * 100).round().astype('int8')
    nitrate2.name = 'nitrogen'
    dairy2 = pd.concat([dairy1['typology']['typology'], phos2, nitrate2], axis=1)


































































































