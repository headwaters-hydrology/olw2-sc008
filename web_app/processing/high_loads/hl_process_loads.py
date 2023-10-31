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

cols_rename = {'ecoli_ratio': 'EC', 'drp_ratio': 'DRP', 'nnn_ratio': 'NO', 'tn_ratio': 'TN', 'tp_ratio': 'TP'}

rec_fields = ['Catchment name', 'Strahler stream order', 'Bedrock', 'Boulder', 'CoarseGravel', 'Cobble', 'FineGravel', 'Mud', 'Sand', '1 in 5 year low flow', 'February flow seasonality', 'FRE3', 'MALF', 'Mean Flow', 'Median flow', 'Month lowest mean flow', '% EPT Richness', 'EPT', 'EPTNoHydrop', 'Invertebrate Taxa Richness', 'MCI (2021)', 'Climate class', 'Geology class', 'Landcover class', 'Network position class', 'Topography class', 'Upstream catchment area', 'Valley landform class', 'Suspended sediment load', 'Ammoniacal nitrogen median', 'ANZECC elevation class', 'CHLA 92%', 'CHLA Mean', 'Dissolved reactive phosphorus median', 'Dissolved reactive phosphorus Q95', 'E. coli G260', 'E. coli G540', 'E. coli median', 'E. coli Q95', 'Nitrate + nitrite median', 'Nitrate + nitrite Q95', 'Temperature', 'Total nitrogen median', 'Total phosphorus median', 'Total suspended solids', 'Turbidity median', 'Visual clarity median']


def hl_process_loads():
    """

    """
    ## >= 4th order segments
    # REC data
    w0 = nzrec.Water(utils.nzrec_data_path)

    stream_orders = {way_id: v['Strahler stream order'] for way_id, v in w0._way_tag.items()}

    ways_4th_up = set([i for i, v in stream_orders.items() if v > 3])

    rec_4th_data_list = []
    for way_id in ways_4th_up:
        # print(way_id)
        tags = w0._way_tag[way_id]
        data = {name: tags[name] for name in rec_fields}
        data['nzsegment'] = way_id
        rec_4th_data_list.append(data)

    rec_4th_data = pd.DataFrame(rec_4th_data_list).set_index('nzsegment')
    rec_4th_data.to_csv('/home/mike/data/OLW/web_app/rivers/rec_4th_order_and_greater.csv')

    # Load results
    hl0 = pd.read_csv(utils.rivers_high_loads_reaches_csv_path)

    hl_data_cols = [col for col in hl0.columns if '_ratio' in col]
    h_cols = ['nzsegment'] + hl_data_cols

    hl1 = hl0[h_cols].rename(columns=cols_rename).copy()
    for col in cols_rename.values():
        hl1[col] = (hl1[col].round(2) * 100).astype('int8')

    hl1['nzsegment'] = hl1['nzsegment'].astype('int32')

    hl2 = hl1.set_index('nzsegment').to_xarray()
    hl2['nzsegment'] = hl2['nzsegment'].astype('int32')

    hdf5tools.xr_to_hdf5(hl2, utils.rivers_high_loads_reaches_path)

    ## Monitoring sites data
    data_list = []
    for file in utils.high_res_moni_dir.iterdir():
        if file.suffix == '.csv':
            if 'TN' in file.name:
                if '0' in file.name:
                    data0 = pd.read_csv(file, usecols=['Date', 'sID', 'npID', 'Q', 'ConcDay', 'FluxDay']).rename(columns={'Date': 'date', 'sID': 'lawa_id', 'npID': 'indicator', 'Q': 'flow', 'ConcDay': 'conc', 'FluxDay': 'load'})
                    data0['lawa_id'] = data0['lawa_id'].str.replace('_NIWA', '').str.replace('_', '-')
                else:
                    data0 = pd.read_csv(file, usecols=[1, 2, 3, 4, 17, 18], header=None)
                    data0.columns = ['date', 'lawa_id', 'indicator', 'flow', 'conc', 'load']
                    data0['lawa_id'] = data0['lawa_id'].str.replace('_NIWA', '').str.replace('_', '-')
            else:
                data0 = pd.read_csv(file, usecols=['Date', 'sID', 'npID', 'Q', 'ConcDay', 'FluxDay']).rename(columns={'Date': 'date', 'sID': 'lawa_id', 'npID': 'indicator', 'Q': 'flow', 'ConcDay': 'conc', 'FluxDay': 'load'})
                data0['lawa_id'] = data0['lawa_id'].str.replace('_NIWA', '').str.replace('_', '-')
            data_list.append(data0)

    loads0 = pd.concat(data_list)
    loads0['date'] = pd.to_datetime(loads0['date'])
    loads0['indicator'] = loads0['indicator'].replace({'NNN': 'NO', 'ECOLI': 'EC'})
    loads1 = loads0.set_index(['indicator', 'lawa_id', 'date']).replace([np.inf, -np.inf], np.nan).dropna()
    utils.df_to_feather(loads1.reset_index(), utils.high_res_moni_feather_path)

    # Calcs
    results_list = []
    for index, data in loads1.groupby(['indicator', 'lawa_id']):
        p90 = data.flow.quantile(0.9)
        load90_sum = data.loc[data.flow >= p90, 'load'].sum()
        load_sum = data.load.sum()
        p90_load_perc = int(round((load90_sum/load_sum) * 100))
        results_list.append([*index, p90_load_perc])

    results0 = pd.DataFrame(results_list, columns=['indicator', 'lawa_id', 'perc_load_above_90_flow'])

    ## Add in the other site identifiers
    site_data = gpd.read_file(utils.river_sites_path)
    results1a = pd.merge(site_data.drop('geometry', axis=1), results0, on='lawa_id')
    results1a[results1a.duplicated(['nzsegment', 'indicator'], keep=False)].to_csv('/home/mike/data/OLW/web_app/rivers/high_res/dup_sites.csv', index=False)

    results2a = results1a.groupby(['indicator', 'nzsegment'])[['perc_load_above_90_flow']].mean().to_xarray()
    results2b = results1a.drop_duplicates(['nzsegment']).set_index('nzsegment')[['lawa_id', 'site_id']].to_xarray()

    results2 = results2a.merge(results2b)

    results2['nzsegment'] = results2['nzsegment'].astype('int32')
    results2['perc_load_above_90_flow'].encoding = {'scale_factor': 1, '_FillValue': -99, 'dtype': 'int8'}

    hdf5tools.xr_to_hdf5(results2, utils.rivers_perc_load_above_90_flow_h5_path)



























































