#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 08:51:20 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
import utils

pd.options.display.max_columns = 10


#############################################
### Rivers


def rec_delin():
    rec_rivers0 = gpd.read_feather(utils.rec_rivers_feather)

    ## Find all outlet reaches
    rec_rivers1 = rec_rivers0[rec_rivers0.stream_order > 2][['nzsegment', 'from_node', 'to_node']].copy()

    end_segs = []

    for i, seg in rec_rivers1.iterrows():
        # print(i)
        seg_bool = rec_rivers1.from_node == seg.to_node
        if not seg_bool.any():
            end_segs.append(seg.nzsegment)

    ## Find all upstream reaches
    reaches = rec.find_upstream(end_segs, rec_streams=rec_rivers0, from_node_col='from_node', to_node_col='to_node')

    ## post-process
    reaches2 = rec_rivers0[['nzsegment', 'stream_order', 'from_node', 'to_node', 'geometry']].merge(reaches['nzsegment'].reset_index(), on='nzsegment')

    reaches2['stream_order'] = reaches2['stream_order'].astype('int8')
    # reaches2['nzsegment'] = reaches2['nzsegment'].astype('int32')
    reaches2['start'] = reaches2['start'].astype('int32')
    # reaches2['from_node'] = reaches2['from_node'].astype('int32')
    # reaches2['to_node'] = reaches2['to_node'].astype('int32')

    reaches2['geometry'] = reaches2['geometry'].simplify(10)

    ## Save
    reaches2.to_feather(utils.rec_delin_file, compression='zstd', compression_level=1)


















































