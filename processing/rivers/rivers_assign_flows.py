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
### Assign flows

extra_end_segs = (3076139,)


def process_flows_rec():
    w0 = nzrec.Water(utils.nzrec_data_path)

    # way = {k: v for k, v in w0._way.items()}
    way_index = {k: v for k, v in w0._way_index.items()}
    # node_way = {k: v for k, v in w0._node_way_index.items()}

    flows = {way_id: v['Median flow'] for way_id, v in w0._way_tag.items()}
    stream_orders = {way_id: v['Strahler stream order'] for way_id, v in w0._way_tag.items()}

    ways_3rd_up = set([i for i, v in stream_orders.items() if v > 2])
    way_index_3rd_up = {way_id: set(v).intersection(ways_3rd_up) for way_id, v in way_index.items() if way_id in ways_3rd_up}

    reaches = booklet.open(utils.river_reach_mapping_path)

    ## Calc diff flows
    up_flows_dict = {}
    for way_id in reaches:
        # if way_id not in extra_end_segs:
        down_ways = set([way_id])
        down_flow = flows[way_id]

        new_ways = set(way_index_3rd_up[way_id])

        if (down_flow is None) or (down_flow == 0):
            up_flows_dict[way_id] = 0
        else:
            up_flows_list = [flows[f] for f in new_ways]
            calc_bool = any([f is None for f in up_flows_list])
            if calc_bool:
                up_flows_dict[way_id] = 0
            else:
                up_flow = sum(up_flows_list)

                diff_flow = round(down_flow - up_flow, 3)
                if diff_flow < 0:
                    up_flows_dict[way_id] = 0
                else:
                    up_flows_dict[way_id] = diff_flow

        while new_ways:
            down_ways.update(new_ways)
            old_ways = copy(new_ways)
            new_ways = set()
            for new_way_id in old_ways:
                up_ways = set(way_index_3rd_up[new_way_id]).difference(down_ways)
                new_ways.update(up_ways)

                down_flow = flows[new_way_id]
                if (down_flow is None) or (down_flow == 0):
                    up_flows_dict[new_way_id] = 0
                else:
                    up_flows_list = [flows[f] for f in up_ways]
                    calc_bool = any([f is None for f in up_flows_list])
                    if calc_bool:
                        up_flows_dict[new_way_id] = 0
                    else:
                        up_flow = sum(up_flows_list)

                        diff_flow = round(down_flow - up_flow, 3)
                        if diff_flow < 0:
                            up_flows_dict[new_way_id] = 0
                        else:
                            up_flows_dict[new_way_id] = diff_flow

    with booklet.open(utils.river_flows_rec_path, 'n', key_serializer='uint4', value_serializer='int4') as f:
        for way_id, flow in up_flows_dict.items():
            f[way_id] = int(flow * 1000)


def process_flows_area():
    catches = booklet.open(utils.river_catch_path)
    # reaches = booklet.open(utils.river_reach_mapping_path)

    # count = 0
    # area_count = 0
    # for way_id in reaches:
    #     up_reaches = catches_minor_dict[way_id]
    #     up_reaches_len = len(up_reaches)
    #     count += up_reaches_len

    #     up_catches = catches[way_id]
    #     up_catches_len = len(up_catches)
    #     area_count += up_catches_len

    #     if up_reaches_len != up_catches_len:
    #         raise ValueError

    # count = 0
    # for way_id in reaches:
    #     other = catches_minor_dict[way_id]
    #     other1 = other.iloc[0]['nzsegment']

    #     up_catches = catches[way_id]
    #     up_catches1 = up_catches.iloc[0]['nzsegment']

    #     if other1 != up_catches1:
    #         count += 1


    ## Calc areas
    up_area_dict = {}
    for way_id in catches:
        up_catches = catches[way_id]
        up_catches['area'] = up_catches.area.round().astype('int32')
        up_catches_dict = up_catches.set_index('nzsegment')['area'].to_dict()

        up_area_dict.update(up_catches_dict)

    with booklet.open(utils.river_flows_area_path, 'n', key_serializer='uint4', value_serializer='int4') as f:
        for way_id, flow in up_area_dict.items():
            f[way_id] = flow



def make_gis_file(output_path):
    """

    """
    flows = booklet.open(utils.river_flows_rec_path)
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





























































































































