#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:23:53 2023

@author: mike
"""
import os
import xarray as xr
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
import hdf5tools
import booklet

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

#################################################
### Process loads

# way_id = 11018765


def process_loads_lakes():
    """

    """
    ## Import flows for load calcs
    flows_list = []
    append = flows_list.append
    with booklet.open(utils.river_flows_rec_path) as f:
        for catch_id, flow in f.items():
            append((catch_id, flow))

    flows = pd.DataFrame(flows_list, columns=['nzsegment', 'flow'])

    ## Conc set 1 - TN, TP, DRP, NNN
    conc0 = pd.read_csv(utils.rivers_conc_csv_path1).set_index('nzsegment')
    cols = [col for col in conc0.columns if ('Median' in col)]
    conc1 = conc0[cols].copy()
    cols1 = [col[7:].split('Median')[0] for col in cols]
    conc1.columns = cols1

    loads1 = pd.merge(flows, conc1, on='nzsegment', how='left')
    loads1.loc[loads1[cols1[0]].isnull(), cols1] = 0
    for col in cols1:
        loads1[col] = loads1[col] * loads1['flow']

    loads1 = loads1.drop('flow', axis=1).set_index('nzsegment')

    ## Calc set 2 - e.coli
    conc0 = pd.read_csv(utils.rivers_conc_csv_path2).set_index('nzsegment')
    cols1 = ['e.coli']
    conc1 = conc0[['CurrentQ50']].copy()
    conc1.columns = cols1
    loads2 = pd.merge(flows, conc1, on='nzsegment', how='left')
    loads2.loc[loads2[cols1[0]].isnull(), cols1] = 0
    for col in cols1:
        loads2[col] = loads2[col] * loads2['flow']

    loads2 = loads2.drop('flow', axis=1).set_index('nzsegment')

    ## Calc set 3 - clarity and turbidity
    conc0 = pd.read_csv(utils.rivers_conc_csv_path3, usecols=['nzsegment', 'cumArea', 'CurrCor_cu'])
    conc0['sediment'] = conc0.CurrCor_cu/conc0.cumArea
    loads3 = pd.merge(flows, conc0[['nzsegment', 'sediment']], on='nzsegment', how='left').set_index('nzsegment')
    loads3.loc[loads3.sediment.isnull(), 'sediment'] = 0
    # loads3['sediment'] = loads3['sediment'] * loads3['flow']

    loads3 = loads3.drop('flow', axis=1)

    ## Combine
    combo1 = pd.concat([loads1, loads2, loads3], axis=1)

    ## Remove upstream loads - Not needed because flows are already processed this way
    # ways = set(combo1.index.values)

    # w0 = nzrec.Water(utils.nzrec_data_path)
    # way = {k: v for k, v in w0._way.items()}
    # way_index = {k: v for k, v in w0._way_index.items()}
    # node_way = {k: v for k, v in w0._node_way_index.items()}

    # load_diff_list = []
    # for way_id in ways:
    #     up_ways1 = utils.get_directly_upstream_ways(way_id, node_way, way, way_index)
    #     up_ways2 = ways.intersection(up_ways1)
    #     data1 = combo1.loc[list(up_ways2)].sum()
    #     load_diff = combo1.loc[way_id] - data1

    ## Convert to web app parameter
    cols2 = combo1.columns
    for param, col in utils.indicator_dict.items():
        combo1[param] = combo1[col].copy()

    combo1 = combo1.drop(cols2, axis=1)

    ## Split by catchment
    with booklet.open(utils.lakes_loads_rec_path, 'n', value_serializer='pickle_zstd', key_serializer='uint4', n_buckets=400) as blt:
        with booklet.open(utils.lakes_reaches_mapping_path) as f:
            for catch_id, reaches in f.items():
                combo2 = combo1.loc[combo1.index.isin(reaches)][utils.indicators['lakes']].copy()
                blt[catch_id] = combo2






























































