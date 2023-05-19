#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:03:34 2023

@author: mike
"""
import io
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from gistools import vector
import os
# import tethysts
import base64
import geobuf
import pathlib
import hdf5plugin
import booklet


######################################################
### Parameters

catch_id = 3076139

base_path = pathlib.Path('/media/nvme1/data/OLW/web_app/output/assets')

land_cover_path = pathlib.Path('/media/nvme1/data/OLW/presentation/3076139_rivers_land_cover_reductions.gpkg')

rivers_flows_path = base_path.joinpath('rivers_flows_rec.blt')
rivers_catch_path = base_path.joinpath('rivers_catchments_minor.blt')
rivers_reach_mapping_path = base_path.joinpath('rivers_reaches_mapping.blt')
rivers_reach_gbuf_path = base_path.joinpath('rivers_reaches.blt')

rivers_reach_error_path = base_path.joinpath('rivers_reaches_error.h5')

red_dict = {'NO': 60, 'DR': 70, 'EC': 40, 'BD': 60}

######################################################
### Functions


def calc_river_reach_reductions(catch_id, plan_file, reduction_col='default_reductions'):
    """
    This assumes that the concentration is the same throughout the entire greater catchment. If it's meant to be different, then each small catchment must be set and multiplied by the area to weight the contribution downstream.
    """
    with booklet.open(rivers_catch_path) as f:
        c1 = f[int(catch_id)]

    with booklet.open(rivers_reach_mapping_path) as f:
        branches = f[int(catch_id)]

    # TODO: Package the flow up by catch_id so that there is less work here
    flows = {}
    with booklet.open(rivers_flows_path) as f:
        for way_id in branches:
            flows[int(way_id)] = f[int(way_id)]

    flows_df = pd.DataFrame.from_dict(flows, orient='index', columns=['flow'])
    flows_df.index.name = 'nzsegment'
    flows_df = flows_df.reset_index()

    plan0 = plan_file[[reduction_col, 'geometry']]
    plan1 = plan0[plan0[reduction_col] > 0].to_crs(2193)

    ## Calc reductions per nzsegment given sparse geometry input
    c2 = plan1.overlay(c1)
    c2['sub_area'] = c2.area

    c2['combo_area'] = c2.groupby('nzsegment')['sub_area'].transform('sum')
    c2['prop_reductions'] = c2[reduction_col]*(c2['sub_area']/c2['combo_area'])
    c3 = c2.groupby('nzsegment')[['prop_reductions', 'sub_area']].sum()

    ## Add in missing areas and assume that they are 0 reductions
    c1['tot_area'] = c1.area

    c4 = pd.merge(c1.drop('geometry', axis=1), c3, on='nzsegment', how='left')
    c4.loc[c4['prop_reductions'].isnull(), ['prop_reductions', 'sub_area']] = 0

    c4['reduction'] = (c4['prop_reductions'] * c4['sub_area'])/c4['tot_area']

    ## Scale the reductions to the flows
    c4 = c4.merge(flows_df, on='nzsegment')

    c4['base_flow'] = c4.flow * 100
    c4['prop_flow'] = c4.flow * c4['reduction']

    c5 = c4[['nzsegment', 'base_flow', 'prop_flow']].set_index('nzsegment').copy()
    c5 = {r: list(v.values()) for r, v in c5.to_dict('index').items()}

    props_index = np.array(list(branches.keys()), dtype='int32')
    props_val = np.zeros(props_index.shape)
    for h, reach in enumerate(branches):
        branch = branches[reach]
        t_area = np.zeros(branch.shape)
        prop_area = t_area.copy()

        for i, b in enumerate(branch):
            if b in c5:
                t_area1, prop_area1 = c5[b]
                t_area[i] = t_area1
                prop_area[i] = prop_area1
            else:
                prop_area[i] = 0

        p1 = (np.sum(prop_area)/np.sum(t_area))
        if p1 < 0:
            props_val[h] = 0
        else:
            props_val[h] = p1

    props = xr.Dataset(data_vars={'reduction': (('reach'), np.round(props_val*100).astype('int8')) # Round to nearest even number
                                  },
                        coords={'reach': props_index}
                        )

    ## Filter out lower stream orders
    # so3 = c1.loc[c1.stream_order > 2, 'nzsegment'].to_numpy()
    # props = props.sel(reach=so3)

    return props



#################################################
### Calcs

lc0 = gpd.read_file(land_cover_path)

red1 = calc_river_reach_reductions(catch_id, lc0)
red2 = red1.to_dataframe().reset_index().rename(columns={'reach': 'nzsegment'})

with booklet.open(rivers_reach_gbuf_path) as f:
    branches = f[int(catch_id)]

geo1 = geobuf.decode(branches)

rivers1 = gpd.GeoDataFrame.from_features(geo1['features'], 4326)

rivers2 = rivers1.merge(red2, on='nzsegment')
rivers2.to_file('/media/nvme1/data/OLW/presentation/example_reductions.gpkg')


x1 = xr.open_dataset(rivers_reach_error_path, engine='h5netcdf')
x2 = x1.sel(nzsegment=3081365).load()

n_samples_dict = {}
for param, conc_perc in red_dict.items():
    x3 = x2.sel(indicator=param, conc_perc=conc_perc)
    ns = x3.where(x3.power >= 80).dropna('n_samples').n_samples.values[0]
    n_samples_dict[param] = ns
























































