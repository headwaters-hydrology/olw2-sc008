#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:00:41 2021

@author: mike
"""
import zstandard as zstd
import codecs
import pickle
import pandas as pd
import numpy as np
# import requests
import xarray as xr
import orjson
# from shapely.geometry import shape, mapping
# import tethysts
import os
# import plotly.graph_objs as go

#####################################
### Parameters

# base_url = 'https://api-int.tethys-ts.xyz/tethys/data/'
# base_url = 'http://tethys-api-int:80/tethys/data/'
# base_url = 'https://api.tethys-ts.xyz/tethys/data/'
# base_url = 'http://tethys-api-ext:80/tethys/data/'


#####################################
### Functions


def encode_obj(obj):
    """

    """
    cctx = zstd.ZstdCompressor(level=1)
    c_obj = codecs.encode(cctx.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)), encoding="base64").decode()

    return c_obj


def decode_obj(str_obj):
    """

    """
    dctx = zstd.ZstdDecompressor()
    obj1 = dctx.decompress(codecs.decode(str_obj.encode(), encoding="base64"))
    d1 = pickle.loads(obj1)

    return d1


# def get_stations(base_url, dataset_id):
#     """

#     """
#     resp_fn_stns = requests.post(base_url + 'get_stations', params={'dataset_id': dataset_id}, headers={'Accept-Encoding': 'br'})

#     if not resp_fn_stns.ok:
#         raise ValueError(resp_fn_stns.raise_for_status())

#     fn_stns = orjson.loads(resp_fn_stns.content)

#     return fn_stns


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


# def get_results(tethys, station_id, from_date=None, to_date=None):
#     """

#     """
#     data_list = []
#     for param, ds_id in dataset_ids.items():
#         params = {'dataset_id': ds_id, 'station_ids': station_id, 'squeeze_dims': True}

#         if from_date is not None:
#             params['from_date'] = pd.Timestamp(from_date).isoformat()
#         if to_date is not None:
#             params['to_date'] = pd.Timestamp(to_date).isoformat()

#         data3 = tethys.get_results(**params)

#         data3['time'] = pd.to_datetime(data3['time'].values) + pd.DateOffset(hours=12)
#         coords = list(data3.coords)
#         if 'geometry' in coords:
#             data3 = data3.drop('geometry')
#         if 'height' in coords:
#             data3 = data3.drop('height')

#         data3 = data3[[param]]
#         data_list.append(data3)

#     data = tethysts.utils.xr_concat(data_list).dropna('time')

#     # data = calc_hs(data, hsc_dict)

#     return data


# def get_results(station_id, from_date=None, to_date=None):
#     """

#     """
#     data_list = []
#     for param, ds_id in dataset_ids.items():
#         params = {'dataset_id': ds_id, 'station_id': station_id, 'squeeze_dims': True}

#         if from_date is not None:
#             params['from_date'] = pd.Timestamp(from_date).isoformat()
#         if to_date is not None:
#             params['to_date'] = pd.Timestamp(to_date).isoformat()

#         resp_fn_results = requests.get(base_url + 'get_results', params=params, headers={'Accept-Encoding': 'br'})

#         if not resp_fn_results.ok:
#             raise ValueError(resp_fn_results.raise_for_status())

#         fn_results = orjson.loads(resp_fn_results.content)

#         data3 = xr.Dataset.from_dict(fn_results)

#         data3['time'] = pd.to_datetime(data3['time'].values) + pd.DateOffset(hours=12)
#         coords = list(data3.coords)
#         if 'geometry' in coords:
#             data3 = data3.drop('geometry')
#         if 'height' in coords:
#             data3 = data3.drop('height')

#         data3 = data3[[param]]
#         data_list.append(data3)

#     data = tethysts.utils.xr_concat(data_list).dropna('time')

#     data = calc_hs(data, hsc_dict)

#     return data



# def stns_dict_to_gdf(stns):
#     """

#     """
#     stns1 = copy.deepcopy(stns)
#     geo1 = [shapely.geometry.Point(s['geometry']['coordinates']) for s in stns1]

#     [s.update({'from_date': s['time_range']['from_date'], 'to_date': s['time_range']['to_date']}) for s in stns1]
#     for s in stns1:
#         _ = s.pop('geometry')
#         _ = s.pop('time_range')
#         if 'stats' in s:
#             _ = s.pop('stats')

#     df1 = pd.DataFrame(stns1)
#     df1['from_date'] = pd.to_datetime(df1['from_date'])
#     df1['to_date'] = pd.to_datetime(df1['to_date'])
#     df1['modified_date'] = pd.to_datetime(df1['modified_date'])

#     stns_gpd1 = gpd.GeoDataFrame(df1, crs=4326, geometry=geo1)

#     return stns_gpd1


# def render_plot(results, hsc_dict, species, refs):
#     """

#     """
#     results1 = calc_hs(results, hsc_dict, species, refs)

#     fig = go.Figure()

#     times = pd.to_datetime(results1['time'].values)

#     for s in refs:
#         # if 'name' in grp:
#         #     name = str(grp['name'].values)
#         #     showlegend = True
#         # elif 'ref' in grp:
#         #     name = str(grp['ref'].values)
#         #     showlegend = True
#         # else:
#         #     name = None
#         #     showlegend = False

#         fig.add_trace(
#             go.Scattergl(
#             x=times,
#             y=results1[s].values,
#             mode='markers',
#             showlegend=True,
#             name=s,
#             opacity=0.8)
#             )

#     layout = dict(paper_bgcolor = '#F4F4F8', plot_bgcolor = '#F4F4F8', showlegend=True, height=780, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(l=20, r=20, t=20, b=20), yaxis_title='Habitat Suitability Index')

#     fig.update_layout(**layout)
#     fig.update_xaxes(
#         type='date',
#         # range=[from_date.date(), to_date.date()],
#         # rangeslider=dict(visible=True),
#         # rangeslider_range=[from_date, to_date],
#         # rangeslider_visible=True,
#         rangeselector=dict(
#             buttons=list([
#                 dict(step="all", label='1y'),
#                 # dict(count=1, label="1 year", step="year", stepmode="backward"),
#                 dict(count=6, label="6m", step="month", stepmode="backward"),
#                 dict(count=1, label="1m", step="month", stepmode="backward"),
#                 dict(count=7, label="7d", step="day", stepmode="backward")
#                 ])
#             )
#         )

#     fig.update_yaxes(
#         # autorange = True,
#         # fixedrange= False,
#         range=[-0.01, 1.01])

#     return fig


# def stn_labels(stns, init_active):
#     """

#     """
#     stns_name_list = []
#     append = stns_name_list.append
#     for stn_id, stn in stns[init_active].items():
#         name_dict = {'label': stn['ref'], 'value': stn['station_id']}
#         append(name_dict)

#     return stns_name_list


# def stns_to_geojson(stns, init_active):
#     """

#     """
#     gj = {'type': 'FeatureCollection', 'features': []}
#     for s in list(stns[init_active].values()):
#         if s['geometry']['type'] in ['Polygon', 'LineString']:
#             geo1 = shape(s['geometry'])
#             geo2 = mapping(geo1.centroid)
#         else:
#             geo2 = s['geometry']

#         if 'name' in s:
#             sgj = {'type': 'Feature', 'geometry': geo2, 'properties': {'name': s['station_id'], 'tooltip': s['name']}}
#         elif 'ref' in s:
#             sgj = {'type': 'Feature', 'geometry': geo2, 'properties': {'name': s['station_id'], 'tooltip': s['ref']}}
#         else:
#             sgj = {'type': 'Feature', 'geometry': geo2, 'properties': {'name': s['station_id']}}
#         gj['features'].append(sgj)

#     return gj


# def stn_date_range(stn, freq='365D'):
#     """

#     """
#     from_date = pd.Timestamp(stn['time_range']['from_date'])
#     to_date = pd.Timestamp(stn['time_range']['to_date'])
#     to_date1 = to_date.ceil('D')

#     from_date1 = (to_date1 - pd.Timedelta(freq))

#     if from_date1 < from_date:
#         from_date1 = from_date.ceil('D')

#     return from_date1, to_date1
