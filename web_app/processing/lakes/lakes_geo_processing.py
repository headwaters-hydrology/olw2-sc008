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


def lakes_geo_process():
    # lakes0 = xr.open_dataset(utils.lakes_stdev_path, engine='h5netcdf')
    # lakes0['LFENZID'] = lakes0['LFENZID'].astype('int32')

    # lakes1 = lakes0.sel(model='BoostingRegressor', drop=True)['stdev'].to_dataframe().reset_index()

    missing = gpd.read_file(utils.lakes_missing_3rd_path)
    missing_ids = missing.LFENZID.values

    ## Lakes catchments
    fenz_catch0 = gpd.read_file(utils.lakes_fenz_catch_path).rename(columns={'LID': 'LFENZID', 'Name': 'name'})
    fenz_catch0['LFENZID'] = fenz_catch0['LFENZID'].astype('int32')
    fenz_catch0 = fenz_catch0.dropna(subset=['LFENZID', 'name']).copy()
    fenz_catch0['name'] = fenz_catch0['name'].apply(lambda x: ' '.join(x.split()))
    fenz_catch1 = fenz_catch0.loc[fenz_catch0.name != '', ['LFENZID', 'name', 'geometry']].reset_index(drop=True).copy()
    fenz_catch1['geometry'] = fenz_catch1.buffer(0.01).simplify(0)

    new_geos = []
    for g in fenz_catch1['geometry']:
        if g.geom_type != 'Polygon':
            sizes = []
            geoms = list(g.geoms)
            for g0 in geoms:
                sizes.append(g0.area)
            max1 = np.argmax(sizes)
            new_geos.append(geoms[max1])
        else:
            new_geos.append(g)

    fenz_catch1['geometry'] = new_geos

    # Select only lakes/catchments that are within a 3rd order stream
    rec_rivers0 = gpd.read_feather(utils.rec_rivers_feather)
    catch_so1 = gpd.sjoin(fenz_catch1, rec_rivers0, 'left')
    catch_so2 = catch_so1.groupby('LFENZID')['stream_order'].max()
    catch_so3 = catch_so2[catch_so2 >= 3].index.values

    fenz_catch2 = fenz_catch1[fenz_catch1.LFENZID.isin(catch_so3)].copy()

    # Remove the lakes that don't have stdevs
    fenz_catch2 = fenz_catch2[~fenz_catch2.LFENZID.isin(missing_ids)].copy()

    utils.gpd_to_feather(fenz_catch2, utils.lakes_catch_path)

    # fenz_catch2['geometry'] = fenz_catch2.simplify(30)

    # with booklet.open(utils.lakes_catches_major_path, 'n', value_serializer='zstd', key_serializer='uint2', n_buckets=400) as s:
    #     for LFENZID in fenz_catch2.LFENZID:
    #         geo = fenz_catch2[fenz_catch2.LFENZID == LFENZID].to_crs(4326).set_index('LFENZID', drop=False).__geo_interface__
    #         gbuf = geobuf.encode(geo)
    #         s[LFENZID] = gbuf

    ## Lakes polygons
    lakes_poly0 = gpd.read_file(utils.lakes_fenz_poly_path)
    lakes_poly0['geometry'] = lakes_poly0.simplify(20)
    lakes_poly0.loc[lakes_poly0.Name == 'Lake Ototoa', 'LID'] = 50270

    lakes_poly0 = lakes_poly0.rename(columns={'LID': 'LFENZID', 'Name': 'name', 'ResidenceTime': 'residence_time', 'MaxDepth': 'max_depth'})
    lakes_poly0 = lakes_poly0.dropna(subset=['LFENZID', 'name']).copy()
    lakes_poly0['LFENZID'] = lakes_poly0['LFENZID'].astype('int32')
    lakes_poly0.loc[lakes_poly0['residence_time'].isnull(), 'residence_time'] = lakes_poly0['residence_time'].median()
    lakes_poly0.loc[lakes_poly0['residence_time'] < 1, 'residence_time'] = 1
    lakes_poly0['residence_time'] = lakes_poly0['residence_time'].round().astype('int32')
    lakes_poly0.loc[lakes_poly0['max_depth'].isnull(), 'max_depth'] = lakes_poly0['max_depth'].median()
    lakes_poly0.loc[lakes_poly0['max_depth'] < 1, 'max_depth'] = 1
    lakes_poly0['max_depth'] = lakes_poly0['max_depth'].round().astype('int32')

    lakes_poly0 = lakes_poly0.drop_duplicates(subset=['LFENZID'])

    lakes_poly1 = lakes_poly0[lakes_poly0.LFENZID.isin(fenz_catch2.LFENZID.values)].copy()

    # lakes_poly1 = lakes_poly0[lakes_poly0.LFENZID.isin(lakes0.LFENZID.values)].copy()
    # lakes_poly1['name'] = lakes_poly1['name'].apply(lambda x: ' '.join(x.split()))
    # lakes_poly1 = lakes_poly1.loc[lakes_poly1.name != '', ['LFENZID', 'name', 'geometry']].copy()
    # lakes_poly1 = lakes_poly1.drop_duplicates(subset=['LFENZID'])

    lakes_poly2 = lakes_poly1[['LFENZID', 'name', 'residence_time', 'max_depth', 'geometry']].reset_index(drop=True).copy()

    utils.gpd_to_feather(lakes_poly2, utils.lakes_poly_path)

    with booklet.open(utils.lakes_poly_gbuf_path, 'n', value_serializer='zstd', key_serializer='uint2', n_buckets=400) as s:
        for LFENZID in lakes_poly2.LFENZID:
            geo = lakes_poly2[lakes_poly2.LFENZID == LFENZID].to_crs(4326).set_index('LFENZID', drop=False).__geo_interface__
            gbuf = geobuf.encode(geo)
            s[LFENZID] = gbuf

    # Point locations
    sites = lakes_poly2.copy()
    sites['geometry'] = sites.geometry.centroid.to_crs(4326)

    sites_geo = sites.set_index('LFENZID').__geo_interface__

    sites_gbuf = geobuf.encode(sites_geo)

    with open(utils.lakes_points_gbuf_path, 'wb') as f:
        f.write(sites_gbuf)

    utils.gpd_to_feather(sites, utils.lakes_points_path)




    ## Error assessments
    # lakes0['CV'] = lakes0.CV.round(3)

    # errors = lakes0['CV'].unique()
    # errors.sort()




















































