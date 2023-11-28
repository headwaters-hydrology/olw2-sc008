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
from shapely.geometry import Point, Polygon, box, LineString, mapping, shape
import orjson

import utils

pd.options.display.max_columns = 10

##################################################
### preprocessing

cat_cols = ['Current5', 'GeomorphicType']
num_cols = ['MaxDepth', 'LakeArea', 'DecTemp', 'DecSolrad', 'Fetch', 'SumWind', 'CatBeech', 'CatGlacial', 'CatHard', 'CatPeat', 'CatPhos', 'CatSlope', 'CatAnnTemp', 'DirectDistCoast', 'ResidenceTime', 'Urban', 'Pasture', 'LakeElevation', 'MeanWind']
model_cols = cat_cols + num_cols


def lakes_geo_process():
    # lakes0 = xr.open_dataset(utils.lakes_stdev_path, engine='h5netcdf')
    # lakes0['LFENZID'] = lakes0['LFENZID'].astype('int32')

    # lakes1 = lakes0.sel(model='BoostingRegressor', drop=True)['stdev'].to_dataframe().reset_index()

    missing = gpd.read_file(utils.lakes_missing_3rd_path)
    missing_ids = missing.LFENZID.values

    lake_ids = pd.read_csv(utils.lakes_stdev_model_input_path).LFENZID.values

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
    fenz_catch2 = fenz_catch2[~fenz_catch2.LFENZID.isin(missing_ids) & fenz_catch2.LFENZID.isin(lake_ids)].copy()

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
    lakes_poly0 = lakes_poly0.dropna(subset=['LFENZID']).copy()
    lakes_poly0['LFENZID'] = lakes_poly0['LFENZID'].astype('int32')
    lakes_poly0.loc[lakes_poly0['residence_time'].isnull(), 'residence_time'] = lakes_poly0['residence_time'].median()
    lakes_poly0.loc[lakes_poly0['residence_time'] < 1, 'residence_time'] = 1
    lakes_poly0['residence_time'] = lakes_poly0['residence_time'].round().astype('int32')
    lakes_poly0.loc[lakes_poly0['max_depth'].isnull(), 'max_depth'] = lakes_poly0['max_depth'].median()
    lakes_poly0.loc[lakes_poly0['max_depth'] < 1, 'max_depth'] = 1
    lakes_poly0['max_depth'] = lakes_poly0['max_depth'].round().astype('int32')
    lakes_poly0.loc[lakes_poly0.name.isnull(), 'name'] = 'No name'

    lakes_poly0 = lakes_poly0.drop_duplicates(subset=['LFENZID'])

    lakes_poly1 = lakes_poly0.loc[lakes_poly0.LFENZID.isin(lake_ids), ['LFENZID', 'name', 'residence_time', 'max_depth', 'geometry']].reset_index(drop=True).copy()

    lakes_poly_3rd = lakes_poly1[lakes_poly1.LFENZID.isin(fenz_catch2.LFENZID.values)].copy()

    lakes_poly_3rd.to_file(utils.lakes_poly_3rd_path, index=False)
    lakes_poly1.to_file(utils.lakes_poly_path, index=False)

    with booklet.open(utils.lakes_poly_gbuf_path, 'n', value_serializer='zstd', key_serializer='uint2', n_buckets=4001) as s:
        for LFENZID in lakes_poly1.LFENZID:
            geo = lakes_poly1[lakes_poly1.LFENZID == LFENZID].to_crs(4326).set_index('LFENZID', drop=False).__geo_interface__
            gbuf = geobuf.encode(geo)
            s[LFENZID] = gbuf

    # with booklet.open(utils.lakes_poly_3rd_gbuf_path, 'n', value_serializer='zstd', key_serializer='uint2', n_buckets=4001) as s:
    #     for LFENZID in lakes_poly_3rd.LFENZID:
    #         geo = lakes_poly1[lakes_poly1.LFENZID == LFENZID].to_crs(4326).set_index('LFENZID', drop=False).__geo_interface__
    #         gbuf = geobuf.encode(geo)
    #         s[LFENZID] = gbuf

    ## Point locations of lakes
    # All lakes
    sites = lakes_poly1.copy()
    sites['geometry'] = sites.geometry.centroid.to_crs(4326)

    sites_geo = sites.set_index('LFENZID').__geo_interface__

    sites_gbuf = geobuf.encode(sites_geo)

    with open(utils.lakes_points_gbuf_path, 'wb') as f:
        f.write(sites_gbuf)

    sites.to_file(utils.lakes_points_path, index=False)

    # 3rd order lakes
    sites = lakes_poly_3rd.copy()
    sites['geometry'] = sites.geometry.centroid.to_crs(4326)

    sites_geo = sites.set_index('LFENZID').__geo_interface__

    sites_gbuf = geobuf.encode(sites_geo)

    with open(utils.lakes_points_3rd_gbuf_path, 'wb') as f:
        f.write(sites_gbuf)

    sites.to_file(utils.lakes_points_3rd_path, index=False)

    ## Point locations of monitoring sites
    stdev0 = pd.read_csv(utils.lakes_stdev_moni_path)
    site_loc0 = pd.read_csv(utils.lakes_raw_moni_data_csv_path, usecols=['LawaSiteID', 'SiteID', 'LFENZID', 'Latitude', 'Longitude',]).rename(columns={'LawaSiteID': 'lawa_id', 'SiteID': 'site_id', 'Latitude': 'lat', 'Longitude': 'lon'})
    site_loc1 = site_loc0.drop_duplicates(subset=['lawa_id'])
    site_loc2 = site_loc1[site_loc1.lawa_id.isin(stdev0.lawa_id.unique())].copy()
    site_loc2['LFENZID'] = site_loc2['LFENZID'].astype('int32')

    site_loc3 = vector.xy_to_gpd(['lawa_id', 'site_id', 'LFENZID'], 'lon', 'lat', site_loc2, 4326)

    # All lakes
    with booklet.open(utils.lakes_moni_sites_gbuf_path, 'n', key_serializer='uint4', value_serializer='zstd', n_buckets=4001) as f:
        for LFENZID in lakes_poly1.LFENZID:
            lake_geo = lakes_poly1[lakes_poly1.LFENZID == LFENZID].to_crs(4326).iloc[0].geometry
            site_loc4 = site_loc3[site_loc3.within(lake_geo)].set_index('site_id', drop=False).rename(columns={'site_id': 'tooltip'}).__geo_interface__
            gbuf = geobuf.encode(site_loc4)
            f[LFENZID] = gbuf

    site_loc3.to_file(utils.lakes_moni_sites_gpkg_path)

    # 3rd order lakes
    # with booklet.open(utils.lakes_moni_sites_3rd_gbuf_path, 'n', key_serializer='uint4', value_serializer='zstd', n_buckets=4001) as f:
    #     for LFENZID in lakes_poly_3rd.LFENZID:
    #         lake_geo = lakes_poly_3rd[lakes_poly_3rd.LFENZID == LFENZID].to_crs(4326).iloc[0].geometry
    #         site_loc4 = site_loc3[site_loc3.within(lake_geo)].set_index('site_id', drop=False).rename(columns={'site_id': 'tooltip'}).__geo_interface__
    #         gbuf = geobuf.encode(site_loc4)
    #         f[LFENZID] = gbuf




















































