#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 16:13:01 2022

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
output_path = '/home/mike/git/hh/olw2-sc008'

public_url = 'https://b2.tethys-ts.xyz/file'

buckets = ['gwrc-env', 'ecan-env-monitoring', 'es-hilltop', 'hrc-env', 'tasman-env', 'trc-env', 'niwa-sos', 'orc-env']

dataset_fields = ['dataset_id', 'feature', 'parameter', 'method', 'owner', 'product_code', 'aggregation_statistic', 'frequency_interval', 'utc_offset', 'units', 'license', 'attribution', 'result_type', 'precision']

ds_csv = 'RC_datasets_in_tethys_{date}.csv.zip'
stn_csv = 'RC_stations_in_tethys_{date}.csv.zip'

############################################
### Summaries

date = pd.Timestamp.now().strftime('%Y-%m-%d')

ds_list = []
stn_list = []
for bucket in buckets:
    print(bucket)
    remote = {'public_url': public_url, 'bucket': bucket, 'version': 4}
    ts = Tethys([remote], cache_path)
    datasets = ts.datasets.copy()
    for dataset in datasets:
        stns = ts.get_stations(dataset['dataset_id'])

        for stn in stns:
            s = {k: stn[k] for k in ['dataset_id', 'station_id', 'ref', 'name', 'altitude', 'modified_date'] if k in stn}
            s['n_values'] = stn['dimensions']['time']
            s['from_date'] = stn['time_range']['from_date']
            s['to_date'] = stn['time_range']['to_date']
            geo = stn['geometry']['coordinates']
            s['lat'] = geo[1]
            s['lon'] = geo[0]
            stn_list.append(s)

        ds = {k: dataset[k] for k in dataset_fields}
        time_range = dataset['time_range']
        ds['from_date'] = time_range['from_date']
        ds['to_date'] = time_range['to_date']
        ds['n_stations'] = len(stns)

        ds_list.append(ds)

ds_df = pd.DataFrame(ds_list).drop_duplicates('dataset_id').set_index('dataset_id')

stn_df = pd.DataFrame(stn_list).drop_duplicates(['dataset_id', 'station_id']).set_index(['dataset_id', 'station_id'])

ds_df.to_csv(os.path.join(output_path, ds_csv.format(date=date)))
stn_df.to_csv(os.path.join(output_path, stn_csv.format(date=date)))

## Summarize the datasets
# summ_keys = ['feature', 'parameter', 'method', 'owner']
# ds_df1 = ds_df[ds_df.product_code == 'quality_controlled_data'].drop_duplicates(subset=summ_keys)
# grp = ds_df1.groupby(summ_keys)

# owners = ds_df1.owner.unique()
# owners.sort()

# params = ds_df1.parameter.unique()
# params.sort()

# methods = ds_df1.method.unique()
# methods.sort()

# features = ds_df1.feature.unique()
# features.sort()

# ds_df2 = ds_df1[ds_df1.method == 'sensor_recording']












































































