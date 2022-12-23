#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:17:00 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import intersection
import hdf5tools
import xarray as xr
# import dbm
import shelflet
# import shelve
import multiprocessing as mp
import concurrent.futures
import geobuf

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing

lakes0 = pd.read_csv(utils.raw_lakes_path)
lakes_poly0 = gpd.read_file(utils.lakes_poly_path)
lakes_poly0['geometry'] = lakes_poly0.simplify(20)

lakes0 = lakes0.rename(columns={'SiteID': 'site_id'})

# Locations
lakes1 = pd.merge(lakes_poly0.drop(['elevation', 'geometry'], axis=1), lakes0, on='site_id').drop('site_id', axis=1)

sites0 = lakes1[['name', 'Latitude', 'Longitude']].drop_duplicates(subset=['name']).sort_values('name')

sites = vector.xy_to_gpd('name', 'Longitude', 'Latitude', sites0, 4326)
sites_geo = sites.set_index('name').__geo_interface__

sites_gbuf = geobuf.encode(sites_geo)

with open(utils.lakes_points_gbuf_path, 'wb') as f:
    f.write(sites_gbuf)

sites.to_file(utils.output_path.joinpath('lake_locations.gpkg'))

with shelflet.open(utils.lakes_poly_gbuf_path, 'n') as s:
    for name in lakes_poly0.name:
        geo = lakes_poly0[lakes_poly0.name == name].set_index('name', drop=False).to_crs(4326).__geo_interface__
        gbuf = geobuf.encode(geo)
        s[name] = gbuf


## Error assessments
lakes0['CV'] = lakes0.CV.round(3)

errors = lakes0['CV'].unique()
errors.sort()




















































