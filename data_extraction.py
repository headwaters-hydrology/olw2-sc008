#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:30:33 2022

@author: mike
"""
from tethysts import Tethys
import yaml
import pandas as pd
import os
import numpy as np

pd.options.display.max_columns = 10

##############################################
### Parameters

cache_path = '/home/mike/cache/tethys'
output_path = '/home/mike/git/hh/olw2-sc008/data'

public_url = 'https://b2.tethys-ts.xyz/file'

buckets = [
    # 'gwrc-env',
    # 'ecan-env-monitoring',
    # 'es-hilltop',
    'hrc-env',
    'tasman-env',
    'trc-env',
    'niwa-sos',
    'orc-env'
    ]

index_cols = ['dataset_id', 'station_id', 'time', 'parameter']

data_csv = '{}_part{:02d}_{}.csv.zip'

############################################
### Summaries

date = pd.Timestamp.now().strftime('%Y-%m-%d')

for bucket in buckets:
    print('-- ' + bucket)
    remote = {'public_url': public_url, 'bucket': bucket, 'version': 4}
    ts = Tethys([remote], cache_path)
    datasets = [ds for ds in ts.datasets if (ds['product_code'] == 'quality_controlled_data') and (ds['method'] == 'sensor_recording')]
    for dataset in datasets:
        ds_id = dataset['dataset_id']
        print(ds_id)
        stns = ts.get_stations(ds_id)
        stn_ids1 = [stn['station_id'] for stn in stns]
        n_arrays = len(stn_ids1)//50 + 1

        stn_ids_arr = np.array_split(stn_ids1, n_arrays)

        for i, stn_ids in enumerate(stn_ids_arr):
            data = ts.get_results(ds_id, stn_ids.tolist(), squeeze_dims=True)
            if 'modified_date' in data:
                data = data.drop(['modified_date'])
            data1 = data.drop(['lon', 'lat', 'height', 'geometry']).to_dataframe().reset_index()
            del data

            data1 = data1.dropna(subset=[dataset['parameter']])
            if 'geometry' in data1:
                data1 = data1.drop('geometry', axis=1)
            data1['time'] = data1['time'] + pd.DateOffset(hours=12)
            data1['dataset_id'] = ds_id
            data2 = data1.set_index(['dataset_id', 'station_id', 'time'])
            del data1

            data3 = data2[[dataset['parameter']]].stack()
            data3.index.names = index_cols
            data3.name = 'result'
            data3 = data3.to_frame().reset_index()
            data3 = pd.merge(data3, data2.drop(dataset['parameter'], axis=1), on=['dataset_id', 'station_id', 'time'])
            if 'quality_code' in data3:
                data3['quality_code'] = pd.to_numeric(data3['quality_code'], errors='coerce', downcast='integer')

            owner = dataset['owner']

            path1 = os.path.join(output_path, owner)
            os.makedirs(path1, exist_ok=True)

            data3.to_csv(os.path.join(path1, data_csv.format(ds_id, i+1, date)), index=False)

            del data3




































