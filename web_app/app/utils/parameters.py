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
rivers_catch_4th_pbf_path = app_base_path.joinpath('rivers_catchments_4th.pbf')

rivers_reach_gbuf_path = assets_path.joinpath('rivers_reaches.blt')
rivers_lc_clean_path = assets_path.joinpath('rivers_catch_lc.blt')
rivers_catch_path = assets_path.joinpath('rivers_catchments_minor.blt')
rivers_reach_mapping_path = assets_path.joinpath('rivers_reaches_mapping.blt')
rivers_sites_path = assets_path.joinpath('rivers_sites_catchments.blt')
rivers_sites_3rd_path = assets_path.joinpath('rivers_sites_catchments_3rd.blt')
rivers_loads_rec_path = assets_path.joinpath('rivers_loads_rec.blt')
rivers_catch_name_path = assets_path.joinpath('rivers_catchments_names.blt')
rivers_marae_path = assets_path.joinpath('rivers_catchments_marae.blt')

rivers_catch_lc_gpkg_str = '{base_url}olw-data/olw-sc008/rivers_land_cover_gpkg/{catch_id}_rivers_land_cover_reductions.gpkg'

## High flow load
rivers_high_loads_reaches_path = assets_path.joinpath('rivers_high_flow_loads.h5')
rivers_perc_load_above_90_flow_moni_path = assets_path.joinpath('rivers_perc_load_above_90_flow.h5')


##  Ecology
eco_power_moni_path = assets_path.joinpath('eco_reaches_power_monitored.h5')
eco_power_catch_path = assets_path.joinpath('eco_reaches_power_modelled.h5')
eco_reach_weights_path = assets_path.joinpath('eco_reaches_weights.h5')
eco_sites_path = assets_path.joinpath('eco_sites_catchments.blt')


## Lakes
# lakes_power_combo_path = assets_path.joinpath('lakes_power_combo.h5')
lakes_power_moni_path = assets_path.joinpath('lakes_power_monitored.h5')
lakes_power_model_path = assets_path.joinpath('lakes_power_modelled.h5')
lakes_reductions_model_path = assets_path.joinpath('lakes_reductions_modelled.h5')

lakes_moni_sites_gbuf_path = assets_path.joinpath('lakes_moni_sites.blt')
lakes_pbf_path = app_base_path.joinpath('lakes_points.pbf')
lakes_3rd_pbf_path = app_base_path.joinpath('lakes_points_3rd.pbf')
lakes_poly_gbuf_path = assets_path.joinpath('lakes_poly.blt')
lakes_catches_major_path = assets_path.joinpath('lakes_catchments_major.blt')
lakes_reach_gbuf_path = assets_path.joinpath('lakes_reaches.blt')
lakes_lc_path = assets_path.joinpath('lakes_catch_lc.blt')
lakes_reaches_mapping_path = assets_path.joinpath('lakes_reaches_mapping.blt')
lakes_catches_minor_path = assets_path.joinpath('lakes_catchments_minor.blt')

lakes_catch_lc_gpkg_str = '{base_url}olw-data/olw-sc008/lakes_land_cover_gpkg/{lake_id}_lakes_land_cover_reductions.gpkg'

lakes_loads_rec_path = assets_path.joinpath('lakes_loads_rec.blt')

lakes_marae_path = assets_path.joinpath('lakes_catchments_marae.blt')

## GW
gw_error_path = assets_path.joinpath('gw_points_error.h5')
gw_points_rc_blt = assets_path.joinpath('gw_points_rc.blt')
rc_bounds_gbuf = app_base_path.joinpath('rc_bounds.pbf')

## Land Cover
lc_url = '{}olw-data/olw-sc008/olw_land_cover_reductions.gpkg'.format(base_data_url)
rivers_red_url = '{}olw-data/olw-sc008/olw_rivers_reductions.csv.zip'.format(base_data_url)

lc_catch_pbf_path = assets_path.joinpath('rivers_catch_lc_pbf.blt')

rivers_lc_param_mapping = {
    'Visual Clarity': 'suspended sediment',
    'E.coli': 'e.coli',
    'Dissolved reactive phosphorus': 'total phosphorus',
    'Ammoniacal nitrogen': 'total nitrogen',
    'Nitrate nitrogen': 'total nitrogen',
    'Total nitrogen': 'total nitrogen',
    'Total phosphorus': 'total phosphorus',
    }

rivers_lc_param_effects = {
    'total nitrogen': ['Total nitrogen', 'Nitrate nitrogen', 'Ammoniacal nitrogen'],
    'total phosphorus': ['Total phosphorus', 'Dissolved reactive phosphorus'],
    'e.coli': ['E.coli'],
    'suspended sediment': ['Visual Clarity']
    }

lakes_lc_param_mapping = {
    # 'E.coli': 'e.coli',
    'Total nitrogen': 'total nitrogen',
    'Total phosphorus': 'total phosphorus',
    'Chlorophyll a': 'e.coli',
    # 'Total Cyanobacteria': 'e.coli',
    'Secchi Depth': 'suspended sediment'
    }

lakes_lc_param_effects = {
    'suspended sediment': ['Secchi Depth'],
    'e.coli': ['Chlorophyll a'],
    'total phosphorus': ['Total phosphorus'],
    'total nitrogen': ['Total nitrogen']
    }

rivers_lc_params = list(set(rivers_lc_param_mapping.values()))
rivers_lc_params.sort()

lakes_lc_params = list(set(lakes_lc_param_mapping.values()))
lakes_lc_params.sort()

### Layout
# map_height = '80vh'
map_height = '95vh' # for stand-alone pages
center = [-41.1157, 172.4759]
zoom = 6

hovercard_width = 300
hovercard_open_delay = 1000

attribution = 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

rivers_freq_mapping = {4: 'quarterly', 12: 'monthly', 26: 'fortnightly', 52: 'weekly', 104: 'biweekly', 364: 'daily'}
rivers_time_periods = [5, 10, 20, 30]

eco_freq_mapping = {1: 'yearly', 4: 'quarterly', 12: 'monthly'}
eco_time_periods = [5, 10, 20, 30]
eco_n_sites = [5, 10, 20, 30]
eco_freq_value_dict = {'mci': 1, 'peri': 12, 'sediment': 12}

eco_freq_data_dict = {}
for ind in eco_freq_value_dict:
    eco_freq_data_dict[ind] = []
    if ind == 'mci':
        eco_freq_data_dict[ind].extend([{'label': 'yearly', 'value': str(1)}])
    else:
        for key, value in eco_freq_mapping.items():
            eco_freq_data_dict[ind].extend([{'label': value, 'value': str(key)}])

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

classes_reductions = [0, 20, 40, 60, 80]
bins_reductions = classes.copy()
bins_reductions.append(101)
colorscale_reductions = ['#edf8fb','#b3cde3','#8c96c6','#8856a7','#810f7c']
ctg_reductions = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}%+".format(classes[-1])]

reduction_ratios = range(0, 101, 10)
red_ratios = np.array(list(reduction_ratios), dtype='int8')

rivers_points_hideout = {'classes': [], 'colorscale': ['#232323'], 'circleOptions': dict(fillOpacity=1, stroke=True, weight=1, color='black', radius=site_point_radius), 'colorProp': 'nzsegment'}

rivers_indicator_dict = {'BD': 'Visual Clarity', 'EC': 'E.coli', 'DRP': 'Dissolved reactive phosphorus', 'NH': 'Ammoniacal nitrogen', 'NO': 'Nitrate nitrogen', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

rivers_reduction_cols = list(rivers_indicator_dict.values())

eco_indicator_dict = {'peri': 'Periphyton', 'mci': 'MCI', 'sediment': 'Deposited fine sediment'}

eco_reduction_cols = list(eco_indicator_dict.values())

eco_reductions_values = np.arange(10, 101, 10)
eco_reductions_options = [{'value': v, 'label': str(v)+'%'} for v in eco_reductions_values]

eco_bins_weights = [0, 10, 30, 101]
colorscale_weights = ['#edf8b1','#7fcdbb','#2c7fb8']

lakes_indicator_dict = {'CHLA': 'Chlorophyll a', 'Secchi': 'Secchi Depth', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

lakes_reduction_cols = list(lakes_indicator_dict.values())

lakes_points_hideout = {'classes': [], 'colorscale': ['#232323'], 'circleOptions': dict(fillOpacity=1, stroke=True, weight=1, color='black', radius=site_point_radius), 'colorProp': 'tooltip'}

gw_points_hideout = {'classes': [], 'colorscale': ['#808080'], 'circleOptions': dict(fillOpacity=1, stroke=False, radius=site_point_radius), 'colorProp': 'tooltip'}

gw_freq_mapping = {1: 'Yearly', 4: 'Quarterly', 12: 'monthly', 26: 'fortnightly', 52: 'weekly'}

gw_indicator_dict = {'Nitrate': 'Nitrate nitrogen'}

gw_reductions_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

gw_reductions_options = [{'value': v, 'label': str(v)+'%'} for v in gw_reductions_values]

gw_time_periods = [5, 10, 20, 30]

hfl_indicator_dict = {'EC': 'E.coli', 'DRP': 'Dissolved reactive phosphorus', 'NO': 'Nitrate nitrogen', 'TN': 'Total nitrogen', 'TP': 'Total phosphorus'}

hfl_reduction_cols = list(hfl_indicator_dict.values())


### Improvements Slider marks
marks = []
for i in range(0, 101, 10):
    if (i % 20) == 0:
        marks.append({'label': str(i) + '%', 'value': i})
    else:
        marks.append({'value': i})


### high flow colobar
hfl_colorscale = ['#fef0d9','#fdcc8a','#fc8d59','#e34a33','#b30000']
hfl_bins_weights = [0, 20, 40, 60, 80, 101]
hfl_ctg = ["{}%+".format(weights) for i, weights in enumerate(hfl_bins_weights[:-1])]















