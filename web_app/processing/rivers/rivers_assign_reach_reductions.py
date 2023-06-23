#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:58:09 2023

@author: mike
"""
import booklet
import pandas as pd

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

#############################################
### Parameters

# catch_id = 3076139

params = ['Total nitrogen', 'Total phosphorus']

output_path = '/media/nvme1/data/OLW/web_app/output/rec_segment_reductions.csv'

#############################################
### Functions


def calc_river_reach_reductions(catch_id, reductions, reduction_cols):
    """

    """
    with booklet.open(utils.river_catch_path) as f:
        c1 = f[int(catch_id)]

    with booklet.open(utils.river_reach_mapping_path) as f:
        branches = f[int(catch_id)]

    # TODO: Package the flow up by catch_id so that there is less work here
    # flows = {}
    # with booklet.open(utils.river_flows_rec_path) as f:
    #     for way_id in branches:
    #         flows[int(way_id)] = f[int(way_id)]

    # flows_df = pd.DataFrame.from_dict(flows, orient='index', columns=['flow'])
    # flows_df.index.name = 'nzsegment'
    # flows_df = flows_df.reset_index()

    plan1 = reductions[['typology', 'farm_type', 'land_cover'] + reduction_cols + ['geometry']]
    # plan1 = plan0.to_crs(2193)

    ## Calc reductions per nzsegment given sparse geometry input
    c2 = plan1.overlay(c1)
    c2['sub_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')

    c2b = c2.copy()

    results_list = []
    for col in reduction_cols:
        c2b['prop_reductions'] = c2b[col]*(c2b['sub_area']/c2['combo_area'])
        c3 = c2b.groupby('nzsegment')[['prop_reductions', 'sub_area']].sum()

        ## Add in missing areas and assume that they are 0 reductions
        c1['tot_area'] = c1.area

        c4 = pd.merge(c1.drop('geometry', axis=1), c3, on='nzsegment', how='left')
        c4.loc[c4['prop_reductions'].isnull(), ['prop_reductions', 'sub_area']] = 0

        c4['reduction'] = (c4['prop_reductions'] * c4['sub_area'])/c4['tot_area']

        c5 = c4[['nzsegment', 'reduction']].rename(columns={'reduction': col}).groupby('nzsegment').sum().round(2)
        results_list.append(c5)

    results = pd.concat(results_list, axis=1)

    lc0 = c2.sort_values('combo_area').groupby('nzsegment')[['typology', 'farm_type', 'land_cover']].last()

    results2 = pd.concat([lc0, results], axis=1)
    results2.loc[results2.typology.isnull(), ['typology', 'farm_type', 'land_cover']] = 'Native Forest'

    return results2



############################################
### Processing

reach_red_list = []
with booklet.open(utils.catch_lc_path) as f:
    for catch_id in f:
        print(catch_id)
        results = calc_river_reach_reductions(catch_id, f[catch_id], params)
        reach_red_list.append(results)

reach_red0 = pd.concat(reach_red_list).reset_index()
reach_red1 = reach_red0.groupby('nzsegment')[params].mean().round(2)
reach_red2 = reach_red0.groupby('nzsegment')[['typology', 'farm_type', 'land_cover']].first()

reach_red3 = pd.concat([reach_red2, reach_red1], axis=1)

reach_red3.to_csv(output_path)



































































