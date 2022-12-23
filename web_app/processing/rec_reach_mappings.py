#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:40:22 2022

@author: mike
"""
import utils
from gistools import vector, rec
import shelflet

############################################
### Functions


def reach_mapping():
    reaches2 = utils.read_pkl_zstd(utils.output_path.joinpath(utils.rec_delin_file), True)

    grp1 = reaches2.groupby('start')

    with shelflet.open(utils.river_reach_mapping_path, 'n') as mapping:
        for catch_id, reaches in grp1:
            print(catch_id)

            up1 = rec.find_upstream(reaches.nzsegment.tolist(), reaches, from_node_col='from_node', to_node_col='to_node')

            branches = {}
            for grp, segs in up1['nzsegment'].reset_index().groupby('start')['nzsegment']:
                branches[grp] = segs.values.astype('int32')

            mapping[str(catch_id)] = branches
            mapping.sync()






















































































