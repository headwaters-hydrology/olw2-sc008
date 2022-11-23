#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:22:29 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
import utils

pd.options.display.max_columns = 10


############################################
### Catchment agg

def catch_agg():
    rec_catch0 = gpd.read_file(utils.rec_catch_shp)

    reaches2 = utils.read_pkl_zstd(utils.output_path.joinpath(utils.rec_delin_file), True)

    # rec_catch0['geometry'] = rec_catch0['geometry'].simplify(10)

    ## Extract associated catchments
    rec_catch2 = rec.extract_catch(reaches2.set_index('start').drop('geometry', axis=1), rec_catch=rec_catch0)

    ## Aggregate individual catchments
    rec_shed = rec.agg_catch(rec_catch2)
    rec_shed.columns = ['nzsegment', 'geometry', 'area']

    ## Simplify
    rec_shed['geometry'] = rec_shed['geometry'].simplify(20)

    # rec_catch3 = rec_catch2.copy()
    # rec_catch3['geometry'] = rec_catch3['geometry'].simplify(10)

    ## Save
    utils.write_pkl_zstd(rec_shed.drop('area', axis=1), utils.output_path.joinpath(utils.major_catch_file))
    # utils.write_pkl_zstd(rec_catch3, utils.output_path.joinpath(utils.catch_simple_file))
    utils.write_pkl_zstd(rec_catch2, utils.output_path.joinpath(utils.catch_file))




















































