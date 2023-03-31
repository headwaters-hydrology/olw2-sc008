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

base_path = pathlib.Path('/media/nvme1/data/OLW')

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

start_date = '2021-01-01 00:00:00'
end_date = '2023-01-01 00:00:00'
# end_date = '2021-01-02 00:00:00'

times = pd.date_range(start_date, end_date, freq='3M')

times = times.append(pd.to_datetime([end_date]))
times = times.insert(0, pd.to_datetime(start_date))

output_str = '{site_id}_flow.csv'

############################################
### Queries

sites_resp = requests.post(url=url, json={'query': sites_query}, headers=headers)

sites_list = sites_resp.json()['data']['getObservations']

fail_list = []
obs_list = []
for site in sites_list:
    site_id = site['locationId']
    print(site_id)

    for i, start in enumerate(times[:-1]):
        end = times[i+1]

        obs_query = obs_query_str % (site_id, obs_type, start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S'))

        obs_resp = requests.post(url=url, json={'query': obs_query}, headers=headers)

        if obs_resp.ok:
            obs_resp_list = obs_resp.json()['data']['getObservations'][0]['observations']
            if len(obs_list) > 0:
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








































































