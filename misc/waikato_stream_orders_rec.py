#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:44:10 2023

@author: mike
"""
import nzrec


######################################################
### Parameters

downstream_way_id = 3050201
taupo_way_id = 3136016


######################################################
### Waikato segments

w0 = nzrec.Water('/home/mike/git/nzrec/data')

all_up0 = w0.add_way(downstream_way_id)

all_up1 = all_up0.upstream()
all_up2 = all_up1.ways.copy()
all_up_geo = all_up1.to_gpd()

taupo_up0 = w0.add_way(taupo_way_id)
taupo_up1 = taupo_up0.upstream()
taupo_up2 = taupo_up1.ways.copy()
# taupo_up_geo = taupo_up1.to_gpd()

diff_up0 = all_up2.difference(taupo_up2)
diff_geo = all_up_geo[all_up_geo.way_id.isin(diff_up0)].to_crs(2193).copy()
diff_geo['length'] = diff_geo.geometry.length

## Get stream orders
stream_orders = {int(way_id): w0._way_tag[way_id]['Strahler stream order'] for way_id in diff_up0}

orders_count = {}
for way_id, sorder in stream_orders.items():
    if sorder in orders_count:
        orders_count[sorder] += 1
    else:
        orders_count[sorder] = 1

order_groups = {}
for way_id, sorder in stream_orders.items():
    if sorder in order_groups:
        order_groups[sorder].add(way_id)
    else:
        order_groups[sorder] = set([way_id])

order_lengths = {}
for sorder, way_ids in order_groups.items():
    length = round(diff_geo.loc[diff_geo.way_id.isin(way_ids), 'length'].sum() * 0.001)
    order_lengths[sorder] = length










