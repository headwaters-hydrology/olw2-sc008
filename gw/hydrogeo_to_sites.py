#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:34:04 2023

@author: mike
"""
import os
import pathlib
import geopandas as gpd
import pandas as pd
from gistools import vector

pd.options.display.max_columns = 10

#######################################################
### Parameters

base_path = pathlib.Path('/media/nvme1/Projects/OLW/GW')

hydro_shp = 'NZ_Hydrogeological_systems.shp'
ts_csv = 'clean_n_with_metadata_to_send.csv'

sites_base_shp = 'sites.shp'
sites_join_shp = 'sites_with_hydrogeo.shp'


######################################################
### Clean the ts data

ts0 = pd.read_csv(base_path.joinpath(ts_csv))

sites0 = ts0[['ref', 'nztm_x_x', 'nztm_y_x', 'nztm_x_y', 'nztm_y_y', 'lat', 'long']].drop_duplicates(subset=['ref']).rename(columns={'long': 'lon'})
# sites0 = ts0[['ref', 'nztm_x_x', 'nztm_y_x', 'nztm_x_y', 'nztm_y_y', 'lat', 'long']].drop_duplicates(subset=['ref']).rename(columns={'nztm_x_y': 'nztmx', 'nztm_y_y': 'nztmy', 'long': 'lon'})

del ts0


sites_nztm1 = sites0.dropna(subset=['nztm_x_x', 'nztm_y_x'])[['ref', 'nztm_x_x', 'nztm_y_x']].rename(columns={'nztm_x_x': 'nztmx', 'nztm_y_x': 'nztmy'})
sites_nztm2 = sites0.dropna(subset=['nztm_x_y', 'nztm_y_y'])[['ref', 'nztm_x_y', 'nztm_y_y']].rename(columns={'nztm_x_y': 'nztmx', 'nztm_y_y': 'nztmy'})

sites_wgs = sites0.dropna(subset=['lat', 'lon'])[['ref', 'lat', 'lon']]

sites_nztm = pd.concat([sites_nztm1, sites_nztm2]).drop_duplicates(subset=['ref'])

sites_wgs_gpd = vector.xy_to_gpd('ref', 'lon', 'lat', sites_wgs, 4326).to_crs(2193)
sites_nztm_gpd = vector.xy_to_gpd('ref', 'nztmx', 'nztmy', sites_nztm, 2193)

sites1 = sites_nztm_gpd.combine_first(sites_wgs_gpd)
sites1.crs = sites_nztm_gpd.crs

sites1.to_file(base_path.joinpath(sites_base_shp))

#################################################
### Do the spatial join

poly = gpd.read_file(base_path.joinpath(hydro_shp))
poly['geometry'] = poly.simplify(1)
cols = list(poly.columns)
_ = cols.remove('geometry')

sites2, poly2 = vector.pts_poly_join(sites1, poly, cols)

sites2.crs = sites_nztm_gpd.crs


sites2.to_file(base_path.joinpath(sites_join_shp))


































































