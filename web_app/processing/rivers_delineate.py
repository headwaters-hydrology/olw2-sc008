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
from copy import copy
import nzrec
import booklet
import orjson
import geobuf

import utils

pd.options.display.max_columns = 10


#############################################
### Rivers


def rec_delin():
    w0 = nzrec.Water(utils.nzrec_data_path)

    stream_orders = {way_id: v['Strahler stream order'] for way_id, v in w0._way_tag.items()}

    ways_3rd = [i for i, v in stream_orders.items() if v > 2]

    way = {k: v for k, v in w0._way.items()}
    way_index = {k: v for k, v in w0._way_index.items()}
    node_way = {k: v for k, v in w0._node_way_index.items()}

    end_segs = []
    append = end_segs.append
    for way_id in ways_3rd:
        down_node = way[way_id][-1]
        if len(node_way[down_node]) == 1:
            append(way_id)

    ## Delineate all catchment reaches for the mappings
    with booklet.open(utils.river_reach_mapping_path, 'n', value_serializer='pickle_zstd', key_serializer='uint4') as reaches:
        for i, way_id in enumerate(end_segs):
            print(way_id)
            all_up_ways = nzrec.utils.find_upstream(way_id, node_way, way, way_index)
            branches = {way_id: np.asarray(list(copy(all_up_ways)), dtype='int32')}
            all_up_ways.remove(way_id)
            for up_way in all_up_ways:
                branches[int(up_way)] = np.asarray(list(nzrec.utils.find_upstream(up_way, node_way, way, way_index)), dtype='int32')

            reaches[way_id] = branches

    ## Delineate overall reaches for the geobuf and catchments
    reach_gbuf_dict = {}
    catches_dict = {}
    for way_id in end_segs:
        print(way_id)
        w1 = w0.add_way(way_id)
        w1 = w1.upstream()
        gjson = orjson.loads(w1.to_gpd().to_json())
        gbuf = geobuf.encode(gjson)
        reach_gbuf_dict[way_id] = gbuf

        c1 = w1.catchments()
        catch1 = c1.to_gpd()
        catches_dict[way_id] = catch1.rename(columns={'way_id': 'nzsegment'})

    # Reach geobufs in blt
    with booklet.open(utils.river_reach_gbuf_path, 'n', key_serializer='uint4', value_serializer='zstd') as f:
        for way_id, gbuf in reach_gbuf_dict.items():
            f[way_id] = gbuf

    # Catchment gpds in blt
    with booklet.open(utils.river_catch_major_path, 'n', key_serializer='uint4', value_serializer='gpd_zstd') as f:
        for way_id, catch in catches_dict.items():
            f[way_id] = catch

    # Catchments geobuf
    rec_shed = pd.concat(list(catches_dict.values()))
    rec_shed['geometry'] = rec_shed.simplify(0.00004)

    gjson = orjson.loads(rec_shed.set_index('nzsegment').to_json())

    with open(utils.assets_path.joinpath('catchments.pbf'), 'wb') as f:
        f.write(geobuf.encode(gjson))

    ## Produce a file grouped by all catchments as geodataframes
    reaches = booklet.open(utils.river_reach_mapping_path)

    rec_catch0 = gpd.read_feather(utils.rec_catch_feather)

    with booklet.open(utils.river_catch_path, 'n', key_serializer='uint4', value_serializer='gpd_zstd') as f:
        for way_id, branches in reaches.items():
            segs = branches[way_id]
            catches1 = rec_catch0[rec_catch0.nzsegment.isin(segs)].copy()
            f[way_id] = catches1


















































