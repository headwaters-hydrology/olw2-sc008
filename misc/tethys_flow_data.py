#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:14:11 2023

@author: mike
"""
import pathlib
from tethysts import Tethys
import pandas as pd
import timeit

pd.options.display.max_columns = 10

############################################
### Parameters

dataset_id = 'e4fe750041a69b8481f18a8d'
station_ids = ['fe62bb02032e8e916648eb89', '292f77d85eae60a73014abf8',
               '269a56566c4a1f7d2282a0c3', '37cf8affe1a04ac2df051fd2',
               '77e22cbd65730599f8661abc', '387a3c7b9c3cfeb9b6a46cf6',
               'e8140dcbad35440b76194e45'
               ]

output_path = '/home/mike/data/OLW/web_app/rivers/Horizons_streamflow_high_res.csv'

cache_path = '/media/data01/cache/temp'
# cache_path = '/home/mike/cache/temp'
version_date = '2022-07-01T00:00:00'

############################################
### Run

t1 = Tethys(cache=cache_path)

stns0 = t1.get_stations(dataset_id)

stn_ids = [stn['station_id'] for stn in stns0]

results_list = []
for stn_id in station_ids:
    print(stn_id)
    results0 = t1.get_results(dataset_id, stn_id, squeeze_dims=True)
    results1 = results0.drop(['height', 'geometry'])[['station_id',  'streamflow']].to_dataframe().reset_index()
    results_list.append(results1)

results = pd.concat(results_list).set_index(['station_id', 'time']).sort_index()

results.to_csv(output_path)




rc = t1._results_chunks[dataset_id]['2022-07-01T00:00:00']
path1 = pathlib.Path('/media/data01/cache/temp/e4fe750041a69b8481f18a8d/20220701000000Z')

data = []
for stn_id in station_ids[:2]:
    path2 = path1.joinpath(stn_id)
    for file in path2.iterdir():
        data.append(file)
































































