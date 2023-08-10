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
import nzrec

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

pd.options.display.max_columns = 10

#################################################
### Process loads


segs = [3087421, 3087553, 3088075, 3088017, 3088076, 3095138]
# way_id = 11018765

tags = ['Climate class', 'Geology class', 'Topography class']


def process_loads():
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

    ### Reference conc - crazy...
    w0 = nzrec.Water(utils.nzrec_data_path)

    class_list = []
    for seg in flows.nzsegment:
        data = w0._way_tag[seg]
        data1 = {'nzsegment': seg}
        data1.update({tag.split(' ')[0]: val for tag, val in data.items() if (tag in tags)})
        class_list.append(data1)

    seg_class0 = pd.DataFrame(class_list)
    seg_class0.loc[seg_class0['Climate'].isnull(), ['Climate', 'Topography', 'Geology']] = 'Other'

    ref_conc30 = pd.read_csv(utils.rivers_ref_conc3_csv_path)

    ref_conc30['Climate'] = ref_conc30.REC.str[:2]
    ref_conc30['Topography'] = ref_conc30.REC.str[2:]

    ref_conc31 = ref_conc30.set_index(['param', 'Climate', 'Topography']).drop('REC', axis=1).stack()
    cols = list(ref_conc31.index.names)
    cols[-1] = 'Geology'
    ref_conc31.index.names = cols
    ref_conc31.name = 'conc'
    ref_conc32 = ref_conc31.unstack(0).reset_index()

    ref_conc00 = pd.merge(seg_class0, ref_conc32, on=['Climate', 'Topography', 'Geology'], how='left')
    ref_conc000 = ref_conc00.loc[~ref_conc00['DRP'].isnull()]

    ref_conc20 = pd.read_csv(utils.rivers_ref_conc2_csv_path)
    ref_conc20['Climate'] = ref_conc20.REC.str[:2]
    ref_conc20['Topography'] = ref_conc20.REC.str[2:]

    ref_conc21 = ref_conc20.set_index(['param', 'Climate', 'Topography']).drop('REC', axis=1)['conc']
    ref_conc22 = ref_conc21.unstack(0).reset_index()

    ref_conc01 = pd.merge(ref_conc00.loc[ref_conc00['DRP'].isnull(), ['nzsegment', 'Climate', 'Topography', 'Geology']], ref_conc22, on=['Climate', 'Topography'], how='left')

    ref_conc23 = ref_conc22.groupby('Climate').mean()
    ref_conc23.loc['Other'] = ref_conc23.mean()

    ref_conc02 = pd.merge(ref_conc01.loc[ref_conc01['DRP'].isnull(), ['nzsegment', 'Climate', 'Topography', 'Geology']], ref_conc23, on=['Climate'], how='left')

    ref_conc000 = ref_conc00.loc[~ref_conc00['DRP'].isnull()]
    ref_conc010 = ref_conc01.loc[~ref_conc01['DRP'].isnull()]

    ref_conc0 = pd.concat([ref_conc000, ref_conc010, ref_conc02]).drop(['Climate', 'Topography', 'Geology'], axis=1).set_index('nzsegment')

    ref_conc0.round(4).to_csv(utils.rivers_ref_conc_csv_path)

    ref_conc0 = ref_conc0.rename(columns={'NO3N': 'NNN', 'SS': 'sediment', 'ECOLI': 'e.coli'})

    cols2 = ref_conc0.columns
    for param, col in utils.indicator_dict.items():
        ref_conc0[param] = ref_conc0[col].copy()

    ref_conc0['Ammoniacal nitrogen'] = ref_conc0['NH4N'].copy()

    ref_conc1 = ref_conc0.drop(cols2, axis=1)

    ref_load0 = pd.merge(flows, ref_conc1, on='nzsegment')
    cols1 = [col for col in ref_load0.columns if col not in ['nzsegment', 'flow']]
    for col in cols1:
        ref_load0[col] = ref_load0[col] * ref_load0['flow']

    ref_load0.drop('flow', axis=1).to_csv(utils.rivers_ref_load_csv_path, index=False)

    ## Convert to web app parameter
    cols2 = combo1.columns
    for param, col in utils.indicator_dict.items():
        combo1[param] = combo1[col].copy()

    combo1 = combo1.drop(cols2, axis=1)

    ## Split by catchment
    with booklet.open(utils.river_loads_rec_path, 'n', value_serializer='pickle_zstd', key_serializer='uint4', n_buckets=1600) as blt:
        with booklet.open(utils.river_reach_mapping_path) as f:
            for catch_id, reaches in f.items():
                combo2 = combo1.loc[reaches[catch_id]].copy()
                blt[catch_id] = combo2






























































