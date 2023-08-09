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

params = list(utils.indicator_dict.keys())

output_path1 = '/home/mike/data/OLW/web_app/output/rec_segment_reductions_v02.csv'
output_path2 = '/home/mike/data/OLW/web_app/output/rec_segment_typology_area_ratios_v02.csv'

#############################################
### Functions


def calc_river_reach_reductions(catch_id, reductions, reduction_cols):
    """

    """
    with booklet.open(utils.river_catch_path) as f:
        catches1 = f[int(catch_id)]

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
    c2 = plan1.overlay(catches1)
    c2['sub_area'] = c2.area

    catches1['total_area'] = catches1.area
    catches2 = catches1.drop('geometry', axis=1)
    c3 = pd.merge(c2.drop('geometry', axis=1), catches2, on='nzsegment')

    c_list = []
    for grp, data in c3.groupby('nzsegment'):
        tot_sub_area = data.sub_area.sum()
        tot_area = data.total_area.iloc[0]

        if (tot_sub_area / tot_area) < 0.99:
            diff_area = tot_area - tot_sub_area
            val = ['Native Forest', 'NA', 'Native Forest'] + [0]*len(reduction_cols) + [grp, diff_area, tot_area]
            c_list.append(val)

    extra1 = pd.DataFrame(c_list, columns=c3.columns)
    c3b = pd.concat([c3, extra1])

    # c2.loc[c2.typology.isnull(), ['typology', 'land_cover']] = 'Native Forest'
    # c2.loc[c2[reduction_cols[0]].isnull(), reduction_cols] = 0

    # c2['total_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')

    c3c = c3b.copy()

    results_list = []
    for col in reduction_cols:
        c3c['reduction'] = c3c[col]*(c3c['sub_area']/c3c['total_area'])
        c4 = c3c.groupby('nzsegment')['reduction'].sum().reset_index()

        c5 = c4[['nzsegment', 'reduction']].rename(columns={'reduction': col}).groupby('nzsegment').sum().round(2)
        results_list.append(c5)

    results = pd.concat(results_list, axis=1)

    lc0 = c2.sort_values('sub_area').groupby('nzsegment')[['typology', 'farm_type', 'land_cover']].last()

    results2 = pd.concat([lc0, results], axis=1)
    # results2.loc[results2.typology.isnull(), ['typology', 'farm_type', 'land_cover']] = 'Native Forest'

    return results2, c3b



############################################
### Processing


def process_river_reach_reductions():
    reach_lu_list = []
    reach_red_list = []
    with booklet.open(utils.catch_lc_path) as f:
        for catch_id in f:
            print(catch_id)
            reductions = f[catch_id]
            results1, results2 = calc_river_reach_reductions(catch_id, reductions, params)
            reach_red_list.append(results1)
            reach_lu_list.append(results2)

    reach_red0 = pd.concat(reach_red_list).reset_index()
    reach_red1 = reach_red0.groupby('nzsegment')[params].mean().round(2)
    reach_red2 = reach_red0.groupby('nzsegment')[['typology', 'farm_type', 'land_cover']].first()

    reach_red3 = pd.concat([reach_red2, reach_red1], axis=1)

    reach_red3.to_csv(output_path1)

    reach_lu0 = pd.concat(reach_lu_list)
    reach_lu0.to_csv(output_path2, index=False)
































































