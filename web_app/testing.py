#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:30:36 2022

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkb, wkt
from shapely.strtree import STRtree
import pickle
import io
from rtree import index
from shapely.ops import unary_union
import geobuf
import base64
import orjson
import zstandard as zstd
import pathlib

pd.options.display.max_columns = 10

##############################################
### Parameters

base_path = '/home/mike/data/OLW/web_app'
# %cd '/home/mike/data/OLW/web_app'


rec_rivers_shp = '/home/mike/data/niwa/rec/River_Lines.shp'
rec_catch_shp ='/home/mike/data/niwa/rec/ca240805-e6be-414e-8a39-579459acac2a2020230-1-1l4v6za.5259g.shp'

test_data_csv = 'NH_Results.csv'
test_tree = 'strtree.pkl'

segment_id_col = 'nzsegment'


#############################################
### Functions


def geojson_to_geobuf(geojson):
    return base64.b64encode(geobuf.encode(geojson)).decode()


def read_pkl_zstd(obj, unpickle=False):
    """
    Deserializer from a pickled object compressed with zstandard.

    Parameters
    ----------
    obj : bytes or str
        Either a bytes object that has been pickled and compressed or a str path to the file object.
    unpickle : bool
        Should the bytes object be unpickled or left as bytes?

    Returns
    -------
    Python object
    """
    if isinstance(obj, str):
        with open(obj, 'rb') as p:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(p) as reader:
                obj1 = reader.read()

    elif isinstance(obj, bytes):
        dctx = zstd.ZstdDecompressor()
        obj1 = dctx.decompress(obj)
    else:
        raise TypeError('obj must either be a str path or a bytes object')

    if unpickle:
        obj1 = pickle.loads(obj1)

    return obj1


def write_pkl_zstd(obj, file_path=None, compress_level=1, pkl_protocol=pickle.HIGHEST_PROTOCOL, retries=3):
    """
    Serializer using pickle and zstandard. Converts any object that can be pickled to a binary object then compresses it using zstandard. Optionally saves the object to disk. If obj is bytes, then it will only be compressed without pickling.

    Parameters
    ----------
    obj : any
        Any pickleable object.
    file_path : None or str
        Either None to return the bytes object or a str path to save it to disk.
    compress_level : int
        zstandard compression level.

    Returns
    -------
    If file_path is None, then it returns the byte object, else None.
    """
    if isinstance(obj, bytes):
        p_obj = obj
    else:
        p_obj = pickle.dumps(obj, protocol=pkl_protocol)

    # counter = retries
    # while True:
    #     try:
    #         cctx = zstd.ZstdCompressor(level=compress_level, write_checksum=True, write_content_size=True)
    #         c_obj = cctx.compress(p_obj)

    #         ## Test
    #         ddtx = zstd.ZstdDecompressor()
    #         _ = ddtx.decompress(c_obj)
    #         break

    #     except Exception as err:
    #         print(str(err))
    #         sleep(2)
    #         counter = counter - 1
    #         if counter <= 0:
    #             raise err

    if isinstance(file_path, (str, pathlib.Path)):
        # source_checksum = zlib.adler32(c_obj)

        # counter = retries
        # while True:
        #     try:
        #         with open(file_path, 'wb') as p:
        #             p.write(c_obj)

        #         ## Test the write
        #         with open(file_path, 'rb') as f:
        #             file_checksum = zlib.adler32(f.read())

        #         if source_checksum == file_checksum:
        #             break
        #         else:
        #             raise ValueError('checksum mismatch...')
        #     except Exception as err:
        #         print(str(err))
        #         sleep(2)
        #         counter = counter - 1
        #         if counter <= 0:
        #             raise err

        with open(file_path, 'wb') as f:
            cctx = zstd.ZstdCompressor(level=compress_level, write_content_size=True)
            with cctx.stream_writer(f, size=len(p_obj)) as compressor:
                compressor.write(p_obj)
    else:
        cctx = zstd.ZstdCompressor(level=compress_level, write_content_size=True)
        c_obj = cctx.compress(p_obj)

        return c_obj

#############################################
### Testing

rec_rivers0 = gpd.read_file(os.path.join(base_path, rec_rivers_shp))
# geo1 = rec_rivers0.geometry.tolist()

rec_catch0 = gpd.read_file(os.path.join(base_path, rec_catch_shp))

test_data0 = pd.read_csv(os.path.join(base_path, test_data_csv))

# tree = STRtree(geo1)

# b1 = [g.wkb for g in geo1]


# file_idx = index.Rtree(os.path.join(base_path, 'rtree'))
# file_idx = index.Rtree('rtree')


rec_rivers1 = rec_rivers0[rec_rivers0.StreamOrde > 2][['nzsegment', 'FROM_NODE', 'TO_NODE']].copy()
rec_rivers1['FROM_NODE'] = rec_rivers1.FROM_NODE.astype('int32')
rec_rivers1['TO_NODE'] = rec_rivers1.TO_NODE.astype('int32')

end_segs = []

for i, seg in rec_rivers1.iterrows():
    # print(i)
    seg_bool = rec_rivers1.FROM_NODE == seg.TO_NODE
    if not seg_bool.any():
        end_segs.append(seg.nzsegment)



### Find all upstream reaches
reaches = rec.find_upstream(end_segs, rec_streams=rec_rivers0)

### Clip reaches to in-between sites if required
# if site_delineate == 'between':
#     reaches1 = reaches.reset_index().copy()
#     reaches2 = reaches1.loc[reaches1[segment_id_col].isin(reaches1.start.unique()), ['start', segment_id_col]]
#     reaches2 = reaches2[reaches2.start != reaches2[segment_id_col]]

#     grp1 = reaches2.groupby('start')

#     for index, r in grp1:
# #            print(index, r)
#         r2 = reaches1[reaches1.start.isin(r[segment_id_col])][segment_id_col].unique()
#         reaches1 = reaches1[~((reaches1.start == index) & (reaches1[segment_id_col].isin(r2)))]

#     reaches = reaches1.set_index('start').copy()

### Extract associated catchments
rec_catch2 = rec.extract_catch(reaches, rec_catch=rec_catch0)

### Aggregate individual catchments
rec_shed = rec.agg_catch(rec_catch2)
rec_shed.columns = [segment_id_col, 'geometry', 'area']
# rec_shed1 = rec_shed.merge(pts_seg.drop('geometry', axis=1), on=segment_id_col)


reaches2 = reaches.loc[reaches.StreamOrde > 2, ['nzsegment']].reset_index().copy()


reaches2 = rec_rivers0[['nzsegment', 'StreamOrde', 'FROM_NODE', 'TO_NODE', 'geometry']].merge(reaches2, on='nzsegment')

reaches2['geometry'] = reaches2['geometry'].simplify(20)
rec_shed['geometry'] = rec_shed['geometry'].simplify(20)

rec_shed = rec_shed.drop('area', axis=1)

rec_shed.to_file('rec_shed.gpkg', driver='GPKG')
reaches2.to_file('rec_reaches.gpkg', driver='GPKG')


reaches3 = reaches2.to_crs(4326)
starts = reaches3.start.unique()

reach_dict = {}

for s in starts:
    df1 = reaches3[reaches3['start'] == s].drop('start', axis=1).set_index('nzsegment')
    gjson = orjson.loads(df1.to_json())
    gbuf = geobuf.encode(gjson)
    reach_dict[s] = gbuf


write_pkl_zstd(reach_dict, 'reach_geobuf.pbf.zst')



rec_shed_gbuf = geobuf.encode(orjson.loads(rec_shed.set_index('nzsegment').to_crs(4326).to_json()))
write_pkl_zstd(rec_shed_gbuf, 'catch_geobuf.pbf.zst')


t1 = read_pkl_zstd('catch_geobuf.pbf.zst')
t1 = read_pkl_zstd('reach_geobuf.pbf.zst', True)





















































