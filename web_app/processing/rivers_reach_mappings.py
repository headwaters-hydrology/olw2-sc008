#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:40:22 2022

@author: mike
"""
import utils
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import booklet
import multiprocessing as mp
import concurrent.futures

############################################
### Functions


# def reach_mapping():
#     reaches2 = gpd.read_feather(utils.rec_delin_file)

#     max_nzsegment = reaches2.nzsegment.max()

#     grp1 = reaches2.groupby('start')

#     mapping_dict = {}
#     for catch_id, reaches in grp1:
#         print(catch_id)

#         up1 = rec.find_upstream(reaches.nzsegment.tolist(), reaches, from_node_col='from_node', to_node_col='to_node')

#         # up1 = up1['nzsegment'].copy()
#         # up1 = up1.astype('int32')
#         grp2 = up1.groupby(level='start')['nzsegment']

#         branches = {}
#         for grp, segs in grp2:
#             if grp <= max_nzsegment:
#                 # print(grp)
#                 branches[grp] = segs.values

#         mapping_dict[catch_id] = branches

#     print('Package up in booklet')
#     with booklet.open(utils.river_reach_mapping_path, 'n', value_serializer='pickle_zstd', key_serializer='uint4') as mapping:
#         for catch_id, branches in mapping_dict.items():
#             # print(catch_id)

#             mapping[catch_id] = branches


def reach_mapping(reaches_list, reaches):
    up1 = rec.find_upstream(reaches_list, reaches, from_node_col='from_node', to_node_col='to_node')

    grp2 = up1.groupby(level='start')['nzsegment']

    branches = {}
    for grp, segs in grp2:
        # print(grp)
        branches[grp] = segs.values

    return catch_id, branches




if __name__ == '__main__':
    reaches2 = gpd.read_feather(utils.rec_delin_file)

    max_nzsegment = reaches2.nzsegment.max()

    grp1 = reaches2.groupby('start')

    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
        futures = []
        for catch_id, reaches in grp1:
            print(catch_id)
            reaches_list = reaches.nzsegment.tolist()
            f = executor.submit(reach_mapping, reaches_list, reaches)
            futures.append(f)

        runs = concurrent.futures.wait(futures)

    combo_list = [r.result() for r in runs[0]]

    print('Package up in booklet')
    with booklet.open(utils.river_reach_mapping_path, 'n', value_serializer='pickle_zstd', key_serializer='uint4') as mapping:
        for catch_id, branches in combo_list:
            # print(catch_id)

            mapping[catch_id] = branches


# up1.reset_index().to_feather('/media/nvme1/data/OLW/web_app/output/test1.feather', compression='zstd', compression_level=1)

# up1 = pd.read_feather('/media/nvme1/data/OLW/web_app/output/test1.feather')













































































