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
import orjson
from tethys_utils import misc

pd.options.display.max_columns = 10

############################################
### Parameters

# dataset_id = 'e4fe750041a69b8481f18a8d'
# station_ids = ['fe62bb02032e8e916648eb89', '292f77d85eae60a73014abf8',
#                '269a56566c4a1f7d2282a0c3', '37cf8affe1a04ac2df051fd2',
#                '77e22cbd65730599f8661abc', '387a3c7b9c3cfeb9b6a46cf6',
#                'e8140dcbad35440b76194e45'
#                ]

# dataset_id = '8137f7465451cf034fe69218' # continuous streamflow
# station_ids = ['30c6caa1901d5948e2bde60d', 'a48dcc61f79364a16fe1d41a']

output_path = pathlib.Path('/home/mike/data/OLW/web_app/rivers/high_res')

cache_path = '/media/data01/cache/temp'
# cache_path = '/home/mike/cache/temp'
# version_date = '2022-07-01T00:00:00'

ds_json_path = output_path.joinpath('dataset_metadata.json')

ds_stn_ids_dict = {
    'e4fe750041a69b8481f18a8d': [
        'fe62bb02032e8e916648eb89',
        '292f77d85eae60a73014abf8',
        '269a56566c4a1f7d2282a0c3',
        '37cf8affe1a04ac2df051fd2',
        '77e22cbd65730599f8661abc',
        '387a3c7b9c3cfeb9b6a46cf6',
        'e8140dcbad35440b76194e45'
                   ],
    'f9e1ef502a333a62eff647b3': [
        'fe62bb02032e8e916648eb89',
        '292f77d85eae60a73014abf8',
        '269a56566c4a1f7d2282a0c3',
        '37cf8affe1a04ac2df051fd2',
        '77e22cbd65730599f8661abc',
        '387a3c7b9c3cfeb9b6a46cf6',
        'e8140dcbad35440b76194e45'
                   ],
    '6daf5b844f462ab4a388ad57': [
        'bd195f9147a0f7ff08ffb734',
        'ce2c4de8c789be0af28cd7b2',
        'b04dca655e135fc9fcbab67e'
        ],
    '2fbfbb92f71b3d05c0b8db51': [
        'bd195f9147a0f7ff08ffb734',
        'ce2c4de8c789be0af28cd7b2',
        'b04dca655e135fc9fcbab67e'
        ],
    '8137f7465451cf034fe69218': [
        '81c91187b46a1b81fc402bca',
        'a868aeb1f4a6adbe65554c72',
        '30c6caa1901d5948e2bde60d',
        '914888b88aa0632523da2ff1',
        'a48dcc61f79364a16fe1d41a',
        'b71c8770e9101848e6262a5d'
        ],
    'a4c015b46889bef3068f6fd4': [
        '81c91187b46a1b81fc402bca',
        'a868aeb1f4a6adbe65554c72',
        '30c6caa1901d5948e2bde60d',
        '914888b88aa0632523da2ff1',
        'a48dcc61f79364a16fe1d41a',
        'b71c8770e9101848e6262a5d'
        ],
    '62e9b6ad15ea448e62af4a53': [
        '801792fd7528ba27e92dcce7'
        ],
    'eb7b2e8f7f62fb030372d6d1': [
        '801792fd7528ba27e92dcce7'
        ],
    }

turb_dss = [
    'f9e1ef502a333a62eff647b3',
    '2fbfbb92f71b3d05c0b8db51',
    'a4c015b46889bef3068f6fd4',
    'eb7b2e8f7f62fb030372d6d1'
    ]

############################################
### Run

t1 = Tethys(cache=cache_path)

## Only the requested dss/stns
ds_dict = {}
for ds_id, stn_ids in ds_stn_ids_dict.items():
    results_list = []
    print('-- dataset_id: ' + ds_id)
    ds = t1._datasets[ds_id]
    parameter = ds['parameter']
    stns0 = t1.get_stations(ds_id)
    base_stn_ids = [stn['station_id'] for stn in stns0]
    for stn_id in stn_ids:
        print(stn_id)
        if stn_id in base_stn_ids:
            results0 = t1.get_results(ds_id, stn_id, squeeze_dims=True)
            results1 = results0.drop(['height', 'geometry'])[['station_id',  parameter]].to_dataframe().reset_index()
            results_list.append(results1)
        else:
            print('station not in dataset')

    results = pd.concat(results_list).set_index(['station_id', 'time']).sort_index()
    results = misc.grp_ts_agg(results.reset_index(), 'station_id', 'time', '15T', 'mean', True)
    output_file_path = output_path.joinpath('{}.csv'.format(ds_id))
    results.to_csv(output_file_path)

    ds_dict[ds_id] = ds

with open(ds_json_path, 'wb') as f:
    f.write(orjson.dumps(ds_dict))







# ds = [ds for ds in t1.datasets if (ds['owner'] == 'Taranaki Regional Council') and (ds['parameter'] == 'streamflow') and (ds['aggregation_statistic'] == 'continuous')][0]

# stns0 = t1.get_stations(dataset_id)

# stn_ids = [stn['station_id'] for stn in stns0]

# results_list = []
# for stn_id in station_ids:
#     print(stn_id)
#     if stn_id in stn_ids:
#         results0 = t1.get_results(dataset_id, stn_id, squeeze_dims=True)
#         results1 = results0.drop(['height', 'geometry'])[['station_id',  'streamflow']].to_dataframe().reset_index()
#         results_list.append(results1)
#     else:
#         print('station not in dataset')

# results = pd.concat(results_list).set_index(['station_id', 'time']).sort_index()

# results.to_csv(output_path)




# rc = t1._results_chunks[dataset_id]['2022-07-01T00:00:00']
# path1 = pathlib.Path('/media/data01/cache/temp/e4fe750041a69b8481f18a8d/20220701000000Z')

# data = []
# for stn_id in station_ids[:2]:
#     path2 = path1.joinpath(stn_id)
#     for file in path2.iterdir():
#         data.append(file)
































































