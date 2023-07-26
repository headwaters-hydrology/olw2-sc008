#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:37:14 2023

@author: mike
"""
import pathlib
import requests
import pandas as pd


###################################################
### Parameters

# base_path = pathlib.Path('/media/nvme1/data/OLW')
base_path = pathlib.Path('/media/nvme1/data/OLW/sensor_recordings/nitrate/ecan')

url = 'https://apis.ecan.govt.nz/waterdata/observations/graphql'

headers = {'Ocp-Apim-Subscription-Key': 'baa14cc68c6b460b85f208b8bbb142c3',
 'Content-Type': 'application/json',
 'Accept-Encoding': 'brotli'}

sites_query = """
query {
	getObservations {
		locationId
		name
		nztmx
		nztmy
        type
		unit
	}
}
"""

obs_query_str = """
query {
	getObservations(filter: { locations: { locationId: "%s" }, observationTypes: %s }) {
		observations(filter: { start: "%s", end: "%s" }) {
			qualityCode
			timestamp
			value
		}
	}
}
"""

obs_type = 'FLOW'

start_date = '2018-07-10 00:00:00'
end_date = '2022-09-14 00:00:00'
# end_date = '2021-01-02 00:00:00'

times = pd.date_range(start_date, end_date, freq='3M')

times = times.append(pd.to_datetime([end_date]))
times = times.insert(0, pd.to_datetime(start_date))

output_str = 'ecan_flow_data_for_nitrate.csv'

# site_names = [
#     'Kaiapoi River u/s Harpers Road',
#     'Hurunui River at SH1 recorder',
#     'Gentleman Smith Stream at Hakatere-Heron Road',
#     'Windermere drain Poplar Rd 1000m nth of Windermere Rd',
#     'Main Drain at Sheffield St',
#     'Kaiapoi River at Mandeville bridge'
#     ]

# site_names_base = [
#     'Hakatere-Heron',
#     'Silverstream',
#     'Mandeville'
#     ]

site_ids = {'SQ34353': '65101',
            'SQ36073': '69004',
            'SQ36244': '1698006',
            'SQ32943': '66415',
            }


############################################
### Queries

sites_resp = requests.post(url=url, json={'query': sites_query}, headers=headers)

sites_list = sites_resp.json()['data']['getObservations']

# sites_list1 = []
# for name in site_names_base:
#     sites = [s for s in sites_list if name.lower() in s['name'].lower()]
#     sites_list1.extend(sites)

sites_list2 = [s for s in sites_list if s['locationId'] in site_ids.values()]

fail_list = []
obs_list = []
for site in sites_list2:
    site_id = site['locationId']
    print(site_id)

    for i, start in enumerate(times[:-1]):
        end = times[i+1]

        obs_query = obs_query_str % (site_id, obs_type, start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S'))

        obs_resp = requests.post(url=url, json={'query': obs_query}, headers=headers)

        if obs_resp.ok:
            obs_resp_list = obs_resp.json()['data']['getObservations'][0]['observations']
            if len(obs_resp_list) > 0:
                obs1 = pd.DataFrame(obs_resp_list)
                obs1['ref'] = site_id

                obs_list.append(obs1)
        else:
            print('site/date failed')
            fail_list.append((site_id, start))


obs2 = pd.concat(obs_list)

obs2 = obs2.rename(columns={'timestamp': 'time', 'qualityCode': 'quality_code', 'value': 'flow_m3_s'})
obs2['quality_code'] = obs2['quality_code'].astype('int16')
obs2['ref'] = obs2['ref'].astype('int32')
obs2['time'] = pd.to_datetime(obs2['time'])

obs3 = obs2.groupby(['ref', 'time']).mean()
obs3['quality_code'] = obs3['quality_code'].astype('int16')

obs3.to_csv(base_path.joinpath(output_str))






































































