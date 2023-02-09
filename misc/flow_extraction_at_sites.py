#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 07:39:50 2023

@author: mike
"""
import os
import pandas as pd
import pathlib

pd.options.display.max_columns = 10

######################################################
### Parameters

base_path = pathlib.Path('/media/nvme1/data/OLW')

flow_data_csv = 'NZRiverMaps_hydro_data_2023-01-09.csv'
sites_csv = 'olw_river_sites.csv'

output_csv = 'hydro_stats_at_river_sites.csv'

#####################################################
### Process data


flow_data = pd.read_csv(base_path.joinpath(flow_data_csv))
sites = pd.read_csv(base_path.joinpath(sites_csv)).dropna()

sites['nzsegment'] = sites['nzsegment'].astype('int32')
flow_data['nzsegment'] = flow_data['nzsegment'].astype('int32')

combo = pd.merge(sites, flow_data, on='nzsegment')
combo['NZTM_Easting'] = combo['NZTM_Easting'].astype('i4')
combo['NZTM_Northing'] = combo['NZTM_Northing'].astype('i4')
combo[['Segment length', '1 in 5 year low flow', 'February flow seasonality', 'FRE3', 'MALF', 'Mean Flow', 'Median flow']] = combo[['Segment length', '1 in 5 year low flow', 'February flow seasonality', 'FRE3', 'MALF', 'Mean Flow', 'Median flow']].round(4)

combo.to_csv(base_path.joinpath(output_csv), index=False)


