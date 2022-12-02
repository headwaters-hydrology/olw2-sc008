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
    rec_rivers0 = gpd.read_file(utils.rec_rivers_shp)

    ## Find all outlet reaches
    rec_rivers1 = rec_rivers0[rec_rivers0.StreamOrde > 2][['nzsegment', 'FROM_NODE', 'TO_NODE']].copy()
    rec_rivers1['FROM_NODE'] = rec_rivers1.FROM_NODE.astype('int32')
    rec_rivers1['TO_NODE'] = rec_rivers1.TO_NODE.astype('int32')

    end_segs = []

    for i, seg in rec_rivers1.iterrows():
        # print(i)
        seg_bool = rec_rivers1.FROM_NODE == seg.TO_NODE
        if not seg_bool.any():
            end_segs.append(seg.nzsegment)

    ## Find all upstream reaches
    reaches = rec.find_upstream(end_segs, rec_streams=rec_rivers0)

    ## post-process
    reaches2 = rec_rivers0[['nzsegment', 'StreamOrde', 'FROM_NODE', 'TO_NODE', 'geometry']].rename(columns={'StreamOrde': 'stream_order', 'FROM_NODE': 'from_node', 'TO_NODE': 'to_node'}).merge(reaches['nzsegment'].reset_index(), on='nzsegment')

    reaches2['stream_order'] = reaches2['stream_order'].astype('int8')
    reaches2['nzsegment'] = reaches2['nzsegment'].astype('int32')
    reaches2['start'] = reaches2['start'].astype('int32')
    reaches2['from_node'] = reaches2['from_node'].astype('int32')
    reaches2['to_node'] = reaches2['to_node'].astype('int32')

    reaches2['geometry'] = reaches2['geometry'].simplify(10)

    ## Save
    utils.write_pkl_zstd(reaches2, utils.output_path.joinpath(utils.rec_delin_file))


















































