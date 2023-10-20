#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:38:28 2023

@author: mike
"""
import os
import pathlib
import numpy as np

#################################################
#### Global parameters

### Paths
assets_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).parent.joinpath('assets')

app_base_path = pathlib.Path('/assets')

base_data_url = 'https://b2.tethys-ts.xyz/file/'

# lc_url = '{}olw-data/olw-sc008/olw_land_cover_reductions.gpkg'.format(base_data_url)
# rivers_red_url = '{}olw-data/olw-sc008/olw_rivers_reductions.csv.zip'.format(base_data_url)

## Rivers
rivers_power_model_path = assets_path.joinpath('rivers_reaches_power_modelled.h5')
rivers_power_moni_path = assets_path.joinpath('rivers_reaches_power_monitored.h5')
rivers_reductions_model_path = assets_path.joinpath('rivers_reductions_modelled.h5')
rivers_catch_pbf_path = app_base_path.joinpath('rivers_catchments.pbf')

rivers_reach_gbuf_path = assets_path.joinpath('rivers_reaches.blt')
rivers_lc_clean_path = assets_path.joinpath('rivers_catch_lc.blt')
rivers_catch_path = assets_path.joinpath('rivers_catchments_minor.blt')
rivers_reach_mapping_path = assets_path.joinpath('rivers_reaches_mapping.blt')
rivers_sites_path = assets_path.joinpath('rivers_sites_catchments.blt')
rivers_loads_rec_path = assets_path.joinpath('rivers_loads_rec.blt')
rivers_catch_name_path = assets_path.joinpath('rivers_catchments_names.blt')
rivers_marae_path = assets_path.joinpath('rivers_catchments_marae.blt')

rivers_catch_lc_gpkg_str = '{base_url}olw-data/olw-sc008/rivers_land_cover_gpkg/{catch_id}_rivers_land_cover_reductions.gpkg'

##  Ecology
eco_power_moni_path = assets_path.joinpath('eco_reaches_power_monitored.h5')
eco_reach_weights_path = assets_path.joinpath('eco_reach_weights.h5')
eco_sites_path = assets_path.joinpath('eco_sites_catchments.blt')


## Lakes
lakes_power_combo_path = assets_path.joinpath('lakes_power_combo.h5')
# lakes_power_moni_path = assets_path.joinpath('lakes_power_monitored.h5')
lakes_reductions_model_path = assets_path.joinpath('lakes_reductions_modelled.h5')

lakes_pbf_path = app_base_path.joinpath('lakes_points.pbf')
lakes_poly_gbuf_path = assets_path.joinpath('lakes_poly.blt')
lakes_catches_major_path = assets_path.joinpath('lakes_catchments_major.blt')
lakes_reach_gbuf_path = assets_path.joinpath('lakes_reaches.blt')
lakes_lc_path = assets_path.joinpath('lakes_catch_lc.blt')
lakes_reaches_mapping_path = assets_path.joinpath('lakes_reaches_mapping.blt')
lakes_catches_minor_path = assets_path.joinpath('lakes_catchments_minor.blt')

lakes_catch_lc_gpkg_str = '{base_url}olw-data/olw-sc008/lakes_land_cover_gpkg/{lake_id}_lakes_land_cover_reductions.gpkg'

lakes_loads_rec_path = assets_path.joinpath('lakes_loads_rec.blt')

## GW
gw_error_path = assets_path.joinpath('gw_points_error.h5')
gw_points_rc_blt = assets_path.joinpath('gw_points_rc.blt')
rc_bounds_gbuf = app_base_path.joinpath('rc_bounds.pbf')

## Land Cover
lc_url = '{}olw-data/olw-sc008/olw_land_cover_reductions.gpkg'.format(base_data_url)
rivers_red_url = '{}olw-data/olw-sc008/olw_rivers_reductions.csv.zip'.format(base_data_url)

lc_catch_pbf_path = assets_path.joinpath('rivers_catch_lc_pbf.blt')


### Layout
map_height = '80vh'
center = [-41.1157, 172.4759]
zoom = 6

attribution = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

rivers_freq_mapping = {4: 'quarterly', 12: 'monthly', 26: 'fortnightly', 52: 'weekly', 104: 'biweekly', 364: 'daily'}
rivers_time_periods = [5, 10, 20, 30]

eco_freq_mapping = {1: 'yearly', 4: 'quarterly', 12: 'monthly'}
eco_time_periods = [5, 10, 20, 30]
eco_n_sites = [5, 10, 20, 30]
eco_freq_dict = {'mci': 1, 'peri': 12, 'sediment': 12}

lakes_freq_mapping = {4: 'quarterly', 12: 'monthly', 26: 'fortnightly', 52: 'weekly', 104: 'biweekly', 364: 'daily'}
lakes_time_periods = [5, 10, 20, 30]

catch_style = {'fillColor': 'grey', 'weight': 2, 'opacity': 1, 'color': 'black', 'fillOpacity': 0.1}
lake_style = {'fillColor': '#A4DCCC', 'weight': 4, 'opacity': 1, 'color': 'black', 'fillOpacity': 1}
reach_style = {'weight': 2, 'opacity': 0.75, 'color': 'grey'}

lc_style = dict(weight=1, opacity=0.7, color='white', dashArray='3', fillOpacity=0.7)

style_power = dict(weight=4, opacity=1, color='white')
classes = [0, 20, 40, 60, 80]
bins = classes.copy()
bins.append(101)
colorscale_power = ['#808080', '#FED976', '#FD8D3C', '#E31A1C', '#800026']
ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}%+".format(classes[-1])]

site_point_radius = 6

reduction_ratios = range(10, 101, 10)
red_ratios = np.array(list(reduction_ratios), dtype='int8')

rivers_points_hideout = {'classes': [], 'colorscale': ['#232323'], 'circleOptions': dict(fillOpacity=1, stroke=True, weight=1, color='black', radius=site_point_radius), 'colorProp': 'nzsegment'}

rivers_indicator_dict = {'BD': 'Visual Clarity', 'EC': 'E.coli', 'DRP': 'Dissolved reactive phosporus', 'NH': 'Ammoniacal nitrogen', 'NO': 'Nitrate', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

rivers_reduction_cols = list(rivers_indicator_dict.values())

eco_indicator_dict = {'peri': 'Periphyton', 'mci': 'MCI', 'sediment': 'Percent deposited sediment'}

eco_reduction_cols = list(eco_indicator_dict.values())

lakes_indicator_dict = {'CHLA': 'Chlorophyll a', 'Secchi': 'Secchi Depth', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

lakes_reduction_cols = list(lakes_indicator_dict.values())

gw_points_hideout = {'classes': [], 'colorscale': ['#808080'], 'circleOptions': dict(fillOpacity=1, stroke=False, radius=site_point_radius), 'colorProp': 'tooltip'}

gw_freq_mapping = {1: 'Yearly', 4: 'Quarterly', 12: 'monthly', 26: 'fortnightly', 52: 'weekly'}

gw_indicator_dict = {'Nitrate': 'Nitrate'}

gw_reductions_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

gw_reductions_options = [{'value': v, 'label': str(v)+'%'} for v in gw_reductions_values]

gw_time_periods = [5, 10, 20, 30]


### Improvements Slider marks
marks = []
for i in range(0, 101, 10):
    if (i % 20) == 0:
        marks.append({'label': str(i) + '%', 'value': i})
    else:
        marks.append({'value': i})





















