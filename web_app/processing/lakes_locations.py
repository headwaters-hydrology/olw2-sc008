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
import booklet
# import shelve
import multiprocessing as mp
import concurrent.futures
import geobuf

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing


def lakes_location_process():
    lakes0 = xr.open_dataset(utils.lakes_stdev_path, engine='h5netcdf')
    lakes0['LFENZID'] = lakes0['LFENZID'].astype('int32')

    # lakes1 = lakes0.sel(model='BoostingRegressor', drop=True)['stdev'].to_dataframe().reset_index()

    lakes_poly0 = gpd.read_file(utils.lakes_poly_path)
    lakes_poly0['geometry'] = lakes_poly0.simplify(20)

    lakes_poly0 = lakes_poly0.rename(columns={'LID': 'LFENZID', 'Name': 'name'})
    lakes_poly0 = lakes_poly0.dropna(subset=['LFENZID']).copy()
    lakes_poly0['LFENZID'] = lakes_poly0['LFENZID'].astype('int32')

    lakes_poly1 = lakes_poly0[lakes_poly0.LFENZID.isin(lakes0.LFENZID.values)].copy()
    lakes_poly1['name'] = lakes_poly1['name'].apply(lambda x: ' '.join(x.split()))
    lakes_poly1 = lakes_poly1[lakes_poly1.name != ''].copy()

    # Locations
    # lakes2 = pd.merge(lakes_poly1[['LFENZID', 'Name', 'geometry']], lakes1, on='LFENZID')

    # sites0 = lakes1[['name', 'Latitude', 'Longitude']].drop_duplicates(subset=['name']).sort_values('name')

    sites = lakes_poly1[['name', 'geometry']].copy()
    sites['geometry'] = sites.geometry.centroid.to_crs(4326)

    sites_geo = sites.set_index('name').__geo_interface__

    sites_gbuf = geobuf.encode(sites_geo)

    with open(utils.lakes_points_gbuf_path, 'wb') as f:
        f.write(sites_gbuf)

    sites.to_file(utils.output_path.joinpath('lake_locations.gpkg'))

    lakes_poly2 = lakes_poly1[['name', 'geometry']].set_index('name', drop=False).copy()

    with booklet.open(utils.lakes_poly_gbuf_path, 'n', value_serializer='zstd', key_serializer='str', n_buckets=800) as s:
        for name in lakes_poly2.index:
            geo = lakes_poly2.loc[[name]].to_crs(4326).__geo_interface__
            gbuf = geobuf.encode(geo)
            s[name] = gbuf


    ## Error assessments
    # lakes0['CV'] = lakes0.CV.round(3)

    # errors = lakes0['CV'].unique()
    # errors.sort()




















































