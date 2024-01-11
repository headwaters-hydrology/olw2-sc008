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
import shapely
from shapely.geometry import Point, Polygon, box, LineString, mapping

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10


#############################################
### Rivers

extra_end_segs = [3076139]


def rec_delin():
    # break_points = gpd.read_file(utils.catch_break_points_gpkg).to_crs(4326)
    w0 = nzrec.Water(utils.nzrec_data_path)

    stream_orders = {way_id: v['Strahler stream order'] for way_id, v in w0._way_tag.items()}

    ways_3rd_up = set([i for i, v in stream_orders.items() if v > 2])

    way = {k: v for k, v in w0._way.items()}
    way_index = {k: v for k, v in w0._way_index.items()}
    node_way = {k: v for k, v in w0._node_way_index.items()}
    catch = {k: v for k, v in w0._catch.items()}

    end_segs = []
    append = end_segs.append
    for way_id in ways_3rd_up:
        down_node = way[way_id][-1]
        if len(node_way[down_node]) == 1:
            append(way_id)

    end_segs.extend(extra_end_segs)

    ## Get the segs associated with the break points and remove he end ones from the end_segs
    # break_segs = set()
    # for i, row in break_points.iterrows():
    #     coords = np.array(row.geometry.coords)[0]
    #     way1 = w0.nearest_way(coords)
    #     break_segs.update(way1.ways)

    # for seg in break_segs:
    #     if seg in end_segs:
    #         end_segs.remove(seg)

    ways_1st_2nd = set([i for i, v in stream_orders.items() if v < 3])

    way_index_3rd_up = {way_id: set(v).intersection(ways_3rd_up) for way_id, v in way_index.items() if way_id in ways_3rd_up}

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

    ## Delineate and aggregate 1st and 2nd order streams to the 3rd
    reaches_dict = {}
    for way_id in end_segs:
        # print(way_id)

        all_up_ways = nzrec.utils.find_upstream(way_id, node_way, way, way_index_3rd_up)
        branches = {way_id: np.asarray(list(all_up_ways), dtype='int32')}
        all_up_ways.remove(way_id)
        for up_way in all_up_ways:
            new_up = nzrec.utils.find_upstream(up_way, node_way, way, way_index_3rd_up)
            branches[up_way] = np.asarray(list(new_up), dtype='int32')

        reaches_dict[int(way_id)] = branches

    ## Delineate all the special subcatchments
    # way_list = []
    # for way_id in break_segs:
    #     way1 = w0.add_way(way_id)
    #     all_up_ways = way1.upstream()
    #     way_list.append(all_up_ways)

    # between_list = nzrec.between(way_list)

    # for way in between_list:
    #     way_id = way.id
    #     all_up_ways = way.ways.copy()
    #     branches = {way_id: np.asarray(list(all_up_ways), dtype='int32')}
    #     all_up_ways.remove(way_id)
    #     for up_way in all_up_ways:
    #         new_up = nzrec.utils.find_upstream(up_way, node_way, way, way_index_3rd_up)
    #         branches[up_way] = np.asarray(list(new_up), dtype='int32')

    #     reaches_dict[int(way_id)] = branches

    ## Delineate the end segments but excluding the 2nd and 1st reaches
    with booklet.open(utils.river_reach_mapping_path, 'n', value_serializer='pickle_zstd', key_serializer='uint4', n_buckets=1607) as reaches:
        for way_id, branches in reaches_dict.items():
            reaches[way_id] = branches

    ## Delineate overall reaches for the geobuf and catchments
    reach_gbuf_dict = {}
    catches_major_dict = {}
    catches_major_4th_dict = {}
    catches_minor_dict = {}
    for way_id in reaches_dict:
        # print(way_id)

        ways_up = reaches_dict[way_id][way_id]

        # Reaches
        geo = []
        for w in ways_up:
            nodes = way[w]
            geo.append(LineString(np.array([w0._node[i] * 0.0000001 for i in nodes])).simplify(0.0005))
        data = [{'nzsegment': int(i)} for i in ways_up]

        gdf = gpd.GeoDataFrame(data, geometry=geo, crs=4326).set_index('nzsegment', drop=False)
        gjson = orjson.loads(gdf.to_json())
        gbuf = geobuf.encode(gjson)
        reach_gbuf_dict[way_id] = gbuf

        # Catchments
        geos = [catch[i] for i in ways_up]
        catch1 = gpd.GeoDataFrame(ways_up, geometry=geos, crs=4326, columns=['nzsegment'])
        catches_minor_dict[way_id] = catch1

        geo = shapely.ops.unary_union(geos).buffer(0.00001)
        catches_major_dict[way_id] = geo

        if stream_orders[way_id] > 3:
            catches_major_4th_dict[way_id] = geo


    # Reach geobufs in blt
    with booklet.open(utils.river_reach_gbuf_path, 'n', key_serializer='uint4', value_serializer='zstd', n_buckets=1607) as f:
        for way_id, gbuf in reach_gbuf_dict.items():
            f[way_id] = gbuf

    # Catchment gpds in blt
    with booklet.open(utils.river_catch_major_path, 'n', key_serializer='uint4', value_serializer='wkb_zstd', n_buckets=1607) as f:
        for way_id, catches in catches_major_dict.items():
            f[way_id] = catches

    # Catchments geobuf
    catch_ids = list(catches_major_dict.keys())
    rec_shed = gpd.GeoDataFrame(catch_ids, geometry=list(catches_major_dict.values()), crs=4326, columns=['nzsegment'])
    rec_shed['geometry'] = rec_shed.simplify(0.0004)

    rec_shed.to_file(utils.river_catch_gpkg_path)

    gjson = orjson.loads(rec_shed.set_index('nzsegment').to_json())

    with open(utils.assets_path.joinpath('rivers_catchments.pbf'), 'wb') as f:
        f.write(geobuf.encode(gjson))

    catch_ids = list(catches_major_4th_dict.keys())
    rec_shed = gpd.GeoDataFrame(catch_ids, geometry=list(catches_major_4th_dict.values()), crs=4326, columns=['nzsegment'])
    rec_shed['geometry'] = rec_shed.simplify(0.0004)

    gjson = orjson.loads(rec_shed.set_index('nzsegment').to_json())

    with open(utils.assets_path.joinpath('rivers_catchments_4th.pbf'), 'wb') as f:
        f.write(geobuf.encode(gjson))

    ## Produce a file grouped by all catchments as geodataframes
    with booklet.open(utils.river_catch_path, 'n', key_serializer='uint4', value_serializer='gpd_zstd', n_buckets=1607) as f:
        for way_id, branches in catches_minor_dict.items():
            f[way_id] = branches.to_crs(2193)

    # reaches = booklet.open(utils.river_reach_mapping_path)

    # rec_catch0 = gpd.read_feather(utils.rec_catch_feather)

    # with booklet.open(utils.river_catch_path, 'n', key_serializer='uint4', value_serializer='gpd_zstd') as f:
    #     for way_id, branches in reaches.items():
    #         segs = branches[way_id]
    #         catches1 = rec_catch0[rec_catch0.nzsegment.isin(segs)].copy()
    #         f[way_id] = catches1

    # catch_list = []
    # with booklet.open(utils.river_catch_path) as c1:
    #     for catch in c1.values():
    #         catch_list.append(catch)

    # catch1 = pd.concat(catch_list)
    # catch1.to_file('/home/mike/data/OLW/web_app/rivers/agg_3rd_order_catches.gpkg')




















































