#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:44:19 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
import booklet
import geobuf
import pickle
import zstandard as zstd
import base64
import nzrec
import shapely
import orjson
from shapely.geometry import Point, LineString
import concurrent.futures
import multiprocessing as mp

import utils

pd.options.display.max_columns = 10


#############################################
### Lakes

def lakes_catch_delin():
    w0 = nzrec.Water(utils.nzrec_data_path)

    stream_orders = {way_id: v['Strahler stream order'] for way_id, v in w0._way_tag.items()}

    ways_3rd_up = set([i for i, v in stream_orders.items() if v > 2])

    way = {k: v for k, v in w0._way.items()}
    way_index = {k: v for k, v in w0._way_index.items()}
    # node = {k: v * 0.0000001 for k, v in w0._node.items()}
    # geom_nodes = [Point(v) for i, v in node.items()]
    # node_ids = [i for i in node.keys()]
    node_way = {k: v for k, v in w0._node_way_index.items()}
    catch = {k: v for k, v in w0._catch.items()}

    ways_1st_2nd = set([i for i, v in stream_orders.items() if v < 3])

    way_index_3rd_up = {way_id: set(v).intersection(ways_3rd_up) for way_id, v in way_index.items() if way_id in ways_3rd_up}

    rec_rivers0 = gpd.read_feather(utils.rec_rivers_feather)
    rec_rivers1 = rec_rivers0[rec_rivers0.nzsegment.isin(list(way_index_3rd_up.keys()))].copy()

    ## Associate the 1st and 2nd order streams with the downstream 3rd order or greater streams
    up_3rd_reaches_dict = {}
    for way_id in ways_1st_2nd:
        new_ways = way_index[way_id]

        for w in new_ways:
            so = stream_orders[w]
            if so > 2:
                ways_up = nzrec.utils.find_upstream(way_id, node_way, way, way_index)
                if w in up_3rd_reaches_dict:
                    up_3rd_reaches_dict[w].update(ways_up)
                else:
                    up_3rd_reaches_dict[w] = ways_up
                break

    ## Aggregate the catchments and reassign
    catch_aggs = {}
    for way_id, branches in up_3rd_reaches_dict.items():
        branches.add(way_id)

        if way_id in catch_aggs:
            catch_aggs[way_id].update(branches)
        else:
            catch_aggs[way_id] = branches

    for way_id, branches in catch_aggs.items():
        geo = shapely.ops.unary_union([catch[i] for i in branches])
        catch[way_id] = geo

    # sindex = shapely.strtree.STRtree(list(catch.values()))
    # sindex_ids = list(catch.keys())

    lakes_poly = gpd.read_feather(utils.lakes_poly_path)

    catches_minor_dict = {}
    segs_dict = {}
    reach_gbuf_dict = {}
    for LFENZID in lakes_poly.LFENZID:
        print(LFENZID)

        geom = lakes_poly[lakes_poly.LFENZID == LFENZID].iloc[0]['geometry']
        query_geom_ids = rec_rivers1[rec_rivers1.intersects(geom)].nzsegment.tolist()

        # end_ways = set()
        # tested_set = set()
        # for way_id in query_geom_ids:
        #     if way_id not in tested_set:
        #         down_ways = find_downstream(way_id, node_way, way)
        #         end_ways.add(down_ways[-1])
        #         tested_set.update(down_ways)

        if query_geom_ids:
            # catch_ids = set()
            branches = set()
            for way_id in query_geom_ids:
                if way_id not in branches:
                    all_up_ways = nzrec.utils.find_upstream(way_id, node_way, way, way_index_3rd_up)
                    branches.update(all_up_ways)

                    # branches.update({way_id: np.asarray(list(all_up_ways), dtype='int32')})
                    # catch_ids.update(list(all_up_ways))
                    # all_up_ways.remove(way_id)
                    # for up_way in all_up_ways:
                    #     new_up = nzrec.utils.find_upstream(up_way, node_way, way, way_index_3rd_up)
                    #     branches[up_way] = np.asarray(list(new_up), dtype='int32')

            catch_ids = list(branches)
            geos = [catch[i] for i in catch_ids]
            catch1 = gpd.GeoDataFrame(catch_ids, geometry=geos, crs=4326, columns=['nzsegment']).to_crs(2193)
            catch1['geometry'] = catch1.buffer(0.001).simplify(10)

            # Reaches
            geo = []
            for w in catch_ids:
                nodes = way[w]
                geo.append(LineString(np.array([w0._node[i] * 0.0000001 for i in nodes])).simplify(0.0005))
            data = [{'nzsegment': int(i)} for i in catch_ids]

            gdf = gpd.GeoDataFrame(data, geometry=geo, crs=4326).set_index('nzsegment', drop=False)
            gjson = orjson.loads(gdf.to_json())
            gbuf = geobuf.encode(gjson)

            # LFENZID, branches, catch1, gbuf = lakes_reaches_processing(LFENZID, query_geom_ids, node_way, way, way_index_3rd_up)
            catches_minor_dict[LFENZID] = catch1
            segs_dict[LFENZID] = np.asarray(list(branches), dtype='int32')
            reach_gbuf_dict[LFENZID] = gbuf

    ## Save results
    with booklet.open(utils.lakes_reaches_mapping_path, 'n', value_serializer='numpy_int4_zstd', key_serializer='uint2', n_buckets=400) as mapping:
        for name in segs_dict:
            mapping[name] = segs_dict[name]

    with booklet.open(utils.lakes_catches_minor_path, 'n', value_serializer='gpd_zstd', key_serializer='uint2', n_buckets=400) as mapping:
        for name in catches_minor_dict:
            mapping[name] = catches_minor_dict[name]

    with booklet.open(utils.lakes_reaches_path, 'n', value_serializer='zstd', key_serializer='uint2', n_buckets=400) as mapping:
        for name in reach_gbuf_dict:
            mapping[name] = reach_gbuf_dict[name]









































































