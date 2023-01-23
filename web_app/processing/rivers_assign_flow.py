#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:51:17 2022

@author: mike
"""
import os
import xarray as xr
from shapely.geometry import Point, Polygon, box, LineString
import geopandas as gpd
import pandas as pd
import numpy as np
import hdf5tools
import booklet
import nzrec
from copy import copy

import utils

pd.options.display.max_columns = 10

#######################################
### Assign conc

def process_rec_flows():
    w0 = nzrec.Water(utils.nzrec_data_path)

    # way = {k: v for k, v in w0._way.items()}
    way_index = {k: v for k, v in w0._way_index.items()}
    # node_way = {k: v for k, v in w0._node_way_index.items()}

    flows = {way_id: v['Median flow'] for way_id, v in w0._way_tag.items()}
    stream_orders = {way_id: v['Strahler stream order'] for way_id, v in w0._way_tag.items()}

    ways_3rd_up = set([i for i, v in stream_orders.items() if v > 2])
    way_index_3rd_up = {way_id: set(v).intersection(ways_3rd_up) for way_id, v in way_index.items() if way_id in ways_3rd_up}

    reaches = booklet.open(utils.river_reach_mapping_path)

    up_flows = {}
    for way_id in reaches:
        down_ways = set([way_id])
        down_flow = flows[way_id]
        if down_flow is None:
            down_flow = 0

        new_ways = set(way_index_3rd_up[way_id])
        up_flow = sum([0 if flows[f] is None else flows[f] for f in new_ways])

        diff_flow = round(down_flow - up_flow, 4)
        up_flows[way_id] = diff_flow

        while new_ways:
            down_ways.update(new_ways)
            old_ways = copy(new_ways)
            new_ways = set()
            for new_way_id in old_ways:
                up_ways = set(way_index_3rd_up[new_way_id]).difference(down_ways)
                new_ways.update(up_ways)

                down_flow = flows[new_way_id]
                if down_flow is None:
                    down_flow = 0
                up_flow = sum([0 if flows[f] is None else flows[f] for f in up_ways])
                diff_flow = round(down_flow - up_flow, 4)
                up_flows[new_way_id] = diff_flow

    with booklet.open(utils.river_flows_rec_path, 'n', key_serializer='uint4', value_serializer='int4') as f:
        for way_id, flow in up_flows.items():
            f[way_id] = int(flow * 10000)



# def process_area_flows():
#     w0 = nzrec.Water(utils.nzrec_data_path)

#     # way = {k: v for k, v in w0._way.items()}
#     way_index = {k: v for k, v in w0._way_index.items()}
#     # node_way = {k: v for k, v in w0._node_way_index.items()}

#     flows = {way_id: v['Median flow'] for way_id, v in w0._way_tag.items()}
#     stream_orders = {way_id: v['Strahler stream order'] for way_id, v in w0._way_tag.items()}

#     ways_3rd_up = set([i for i, v in stream_orders.items() if v > 2])
#     way_index_3rd_up = {way_id: set(v).intersection(ways_3rd_up) for way_id, v in way_index.items() if way_id in ways_3rd_up}

#     reaches = booklet.open(utils.river_reach_mapping_path)

#     up_flows = {}
#     for way_id in reaches:
#         down_ways = set([way_id])
#         down_flow = flows[way_id]
#         if down_flow is None:
#             down_flow = 0

#         new_ways = set(way_index_3rd_up[way_id])
#         up_flow = sum([0 if flows[f] is None else flows[f] for f in new_ways])

#         diff_flow = round(down_flow - up_flow, 4)
#         up_flows[way_id] = diff_flow

#         while new_ways:
#             down_ways.update(new_ways)
#             old_ways = copy(new_ways)
#             new_ways = set()
#             for new_way_id in old_ways:
#                 up_ways = set(way_index_3rd_up[new_way_id]).difference(down_ways)
#                 new_ways.update(up_ways)

#                 down_flow = flows[new_way_id]
#                 if down_flow is None:
#                     down_flow = 0
#                 up_flow = sum([0 if flows[f] is None else flows[f] for f in up_ways])
#                 diff_flow = round(down_flow - up_flow, 4)
#                 up_flows[new_way_id] = diff_flow

#     with booklet.open(utils.river_flows_area_path, 'n', key_serializer='uint4', value_serializer='int4') as f:
#         for way_id, flow in up_flows.items():
#             f[way_id] = int(flow * 10000)



def make_gis_file(output_path):
    """

    """
    flows = booklet.open(utils.river_flows_path)
    w0 = nzrec.Water(utils.nzrec_data_path)
    node = w0._node
    way = w0._way

    data = []
    geo = []
    for way_id, flow in flows.items():
        nodes = way[way_id]
        geo.append(LineString(np.array([node[int(i)] * 0.0000001 for i in nodes])))
        data.append([way_id, flow*0.0001])

    gdf = gpd.GeoDataFrame(data, columns = ['nzsegment', 'flow'], geometry=geo, crs=4326)

    gdf.to_file(output_path)





























































































































