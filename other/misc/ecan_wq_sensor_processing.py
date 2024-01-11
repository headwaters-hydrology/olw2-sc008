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

base_path = pathlib.Path('/media/nvme1/data/OLW/sensor_recordings/nitrate/ecan')

flow_csv = 'ecan_flow_data_for_nitrate.csv'

no3_sensor_csv = 'ecan_nitrate_sensor_recordings.csv'

output_combo_high_file = 'ecan_no3_data_high_res.csv'
output_combo_daily_file = 'ecan_no3_data_daily.csv'

site_ids = {'SQ34353': '65101',
            'SQ36073': '69004',
            'SQ36244': '1698006',
            'SQ32943': '66415',
            }

site_ids_dict = {int(v): k for k, v in site_ids.items()}

#####################################################
### Process data

## Flow
flow0 = pd.read_csv(base_path.joinpath(flow_csv)).drop('quality_code', axis=1)
flow0['time'] = pd.to_datetime(flow0['time']).round('T')

flow1 = flow0.replace({'ref': site_ids_dict}).copy()

## NO3 discrete
# data0 = pd.read_csv(no3_discrete_path, usecols=['time', 'nitrate'])
# data0['time'] = pd.to_datetime(data0['time'], infer_datetime_format=True)
# data1 = data0.dropna()
# data1 = data1.set_index('time')
# data1['nitrate'] = pd.to_numeric(data1['nitrate'])
# data2 = data1.rename(columns={'nitrate': 'nitrate_conc_discrete'})
# data2 = data2.reset_index()
# data2['time'] = data2.time.round('10T')

# nitrate1 = data2.groupby('time').mean()

## NO3 sensor
data0 = pd.read_csv(base_path.joinpath(no3_sensor_csv))
data0['time'] = pd.to_datetime(data0['time']).round('T')

data1 = data0.set_index('time').stack()
data1.name = 'nitrate_conc_sensor'
data1.index.names = ['time', 'ref']

data2 = data1.reset_index().groupby(['ref', 'time']).mean()
nitrate2 = data2.reset_index()

### Interpolate flows at nitrate sensor times
combo1 = pd.merge(flow1, nitrate2, on=['ref', 'time'], how='right')

combo_list = []
grp1 = combo1.groupby(['ref'])

for ref, data in grp1:
    data0 = data.set_index('time')

    data0['flow_m3_s'] = data0.flow_m3_s.interpolate('time', limit_area='inside')
    combo_list.append(data0)

combo2 = pd.concat(combo_list).dropna().set_index('ref', append=True)

combo3 = combo2.unstack(1)

combo3.to_csv(base_path.joinpath(output_combo_high_file))

### Agg to daily
combo_list = []
grp2 = combo2.reset_index().groupby(['ref'])

for ref, data in grp2:
    data0 = data.set_index('time').drop('ref', axis=1)

    data1 = misc.discrete_resample(data0, 'D', 'mean')
    data1['ref'] = ref
    data1.index.name = 'time'

    combo_list.append(data1.reset_index())

combo4 = pd.concat(combo_list).dropna()

combo5 = combo4.set_index(['time', 'ref']).unstack(1)

combo5.to_csv(base_path.joinpath(output_combo_daily_file))







