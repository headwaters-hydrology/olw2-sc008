#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:52:39 2023

@author: mike
"""
import os
import pandas as pd
import pathlib
from tethys_utils import misc

pd.options.display.max_columns = 10

######################################################
### Parameters

base_path = pathlib.Path('/media/nvme1/data/OLW')

es_data_path = base_path.joinpath('sensor_recordings/nitrate/ES')

flow_csv_files = ['1. Aparima at thornbury - Flow (pre 2000).csv', '2. Aparima at thornbury - Flow (2000-2020).csv', '3. Aparima at thornbury - Flow (2020 onwards).csv']

no3_sensor_path = es_data_path.joinpath('OLW_Nitrate_Edited.csv')
no3_discrete_path = es_data_path.joinpath('Aparima at thornbury discrete nitrate data.csv')

output_flow_file = 'es_Aparima_at_thornbury_flow_data.csv'
output_combo_high_file = 'es_Aparima_at_thornbury_no3_data_high_res.csv'
output_combo_daily_file = 'es_Aparima_at_thornbury_no3_data_daily.csv'

#####################################################
### Process data

## Flow
flow_list = []

for file in flow_csv_files:
    print(file)
    data0 = pd.read_csv(es_data_path.joinpath(file), usecols=['time', 'flow'])
    data0['time'] = pd.to_datetime(data0['time'], dayfirst=True, infer_datetime_format=True)
    data1 = data0.dropna()
    data1 = data1.set_index('time')
    data1['flow'] = pd.to_numeric(data1['flow'])
    flow_list.append(data1)

flow0 = pd.concat(flow_list)
flow1 = flow0[~flow0.index.duplicated(keep='last')].copy()
flow2 = flow1.reset_index()
flow2['time'] = flow2.time.round('T')

flow2 = flow2.groupby('time').mean()

flow2.to_csv(es_data_path.joinpath(output_flow_file))

## NO3 discrete
data0 = pd.read_csv(no3_discrete_path, usecols=['time', 'nitrate'])
data0['time'] = pd.to_datetime(data0['time'], infer_datetime_format=True)
data1 = data0.dropna()
data1 = data1.set_index('time')
data1['nitrate'] = pd.to_numeric(data1['nitrate'])
data2 = data1.rename(columns={'nitrate': 'nitrate_conc_discrete'})
data2 = data2.reset_index()
data2['time'] = data2.time.round('10T')

nitrate1 = data2.groupby('time').mean()

## NO3 sensor
data0 = pd.read_csv(no3_sensor_path)
data0['time'] = pd.to_datetime(data0['Date'] + ' ' + data0['Time'], dayfirst=True, infer_datetime_format=True)
data1 = data0[['time', 'OPUS Probe Nitrate (g/m3)']].dropna()
data1 = data1.set_index('time')
data1['OPUS Probe Nitrate (g/m3)'] = pd.to_numeric(data1['OPUS Probe Nitrate (g/m3)'])
data2 = data1.rename(columns={'OPUS Probe Nitrate (g/m3)': 'nitrate_conc_sensor'})
data2 = data2.reset_index()
data2['time'] = data2.time.round('T')

nitrate2 = data2.groupby('time').mean()

### Interpolate flows at nitrate sensor times
combo1 = pd.merge(flow2.reset_index(), nitrate2.reset_index(), on='time', how='right')
combo2 = combo1.set_index('time')
combo2['flow'] = combo2.flow.interpolate('time')

combo3 = pd.merge(combo2.reset_index(), nitrate1.reset_index(), on='time', how='left').set_index('time')

combo3.to_csv(es_data_path.joinpath(output_combo_high_file))

### Agg to daily
combo4 = misc.discrete_resample(combo3, 'D', 'mean')
combo4.index.name = 'time'

nitrate3 = nitrate1.reset_index()
nitrate3['time'] = nitrate3.time.dt.floor('D')
nitrate3 = nitrate3.groupby('time').mean()

combo5 = pd.merge(combo4.drop('nitrate_conc_discrete', axis=1).reset_index(), nitrate3.reset_index(), on='time', how='left').set_index('time')

combo5.to_csv(es_data_path.joinpath(output_combo_daily_file))







