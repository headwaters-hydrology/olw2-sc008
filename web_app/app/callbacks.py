#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:55:43 2021

@author: mike
"""
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from dash import dcc, html, dash_table
import dash_leaflet.express as dlx
import xarray as xr
import pathlib
import os
import hdf5plugin

from .app import app
from . import utils

# from app import app
# import utils

################################################
### Parameters

base_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__))).joinpath('assets')
app_base_path = pathlib.Path('/assets')

reaches_path = 'reaches'
# reach_path_str = '{base_path}/{catch_id}.pbf'

sel_data_h5 = 'selection_data.h5'
catch_reaches_file = 'catch_reach_mapping.pkl.zst'

style = dict(weight=4, opacity=1, color='white')
classes = [0, 20, 40, 60, 80]
bins = classes.copy()
bins.append(100)
colorscale = ['#FFEDA0', '#FEB24C', '#FC4E2A', '#BD0026', '#800026']
ctg = ["{}%+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}%+".format(classes[-1])]
colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=300, height=30, position="bottomleft")

catch_reaches = utils.read_pkl_zstd(str(base_path.joinpath(catch_reaches_file)), True)


################################################
### Callbacks


# @app.callback(Output("catch_map", "children"), [Input("catch_map", "click_feature")])
# def catch_hover(feature):
#     if feature is not None:
#         print(feature['id'])
#         return feature['id']


@app.callback(
    Output('catch_id', 'value'),
    [Input('catch_map', 'click_feature')]
    )
def update_catch_id(feature):
    """

    """
    # print(ds_id)
    catch_id = None
    if feature is not None:
        catch_id = feature['id']

    return catch_id


@app.callback(
        Output('reach_map', 'url'),
        [Input('catch_id', 'value')],
        )
# @cache.memoize()
def update_reaches_map(catch_id):
    url = app_base_path.joinpath(reaches_path).joinpath(str(catch_id) + '.pbf')

    return str(url)


@app.callback(
    Output('reach_map', 'hideout'),
    [Input('catch_id', 'value'), Input('indicator', 'value'), Input('percent_change', 'value'), Input('time_period', 'value'), Input('freq', 'value')],
    )
def update_sel_data(catch_id, indicator, percent_change, time_period, freq):
    """

    """
    if isinstance(catch_id, str) and isinstance(indicator, str) and isinstance(percent_change, int) and isinstance(time_period, int) and isinstance(freq, int):
        x1 = xr.open_dataset(base_path.joinpath(sel_data_h5), engine='h5netcdf')
        reaches = catch_reaches[int(catch_id)]
        x2 = x1['percent_likelihood'].sel(indicator=indicator, frequency=freq, percent_change=percent_change, time_period=time_period, nzsegment=reaches, drop=True).copy().load()
        x1.close()
        del x1
        color_arr = pd.cut(x2.values, bins, labels=colorscale).tolist()

        hideout = {'colorscale': color_arr, 'classes': x2.nzsegment.values.tolist(), 'style': style, 'colorProp': 'nzsegment'}
    else:
        hideout = {}

    return hideout








# @app.callback(
#     Output('refs', 'options'),
#     [Input('species', 'value')],
#     [State('hsc_obj', 'data')]
#     )
# def update_ref_options(species, hsc_obj):
#     """

#     """
#     if species is not None:
#         hsc_dict = utils.decode_obj(hsc_obj)
#         hsc_dict1 = hsc_dict[species]

#         options = [{'label': d, 'value': d} for d in list(hsc_dict1.keys())]
#     else:
#         options = []

#     return options


# @app.callback(
#     Output('result_obj', 'data'),
#     [Input('sites', 'value')],
#     [State('tethys_obj', 'data')],
#     )
# def update_results(stn_id, tethys_obj):
#     """

#     """
#     # print(type(stn_id))
#     if stn_id is not None:
#         tethys = utils.decode_obj(tethys_obj)

#         data = utils.encode_obj(utils.get_results(tethys, stn_id))
#     else:
#         data = ''

#     return data


# @app.callback(Output('plots', 'children'),
#               [Input('plot_tabs', 'value'), Input('result_obj', 'data'),
#                Input('species', 'value'), Input('refs', 'value')],
#               [State('sites', 'value'), State('hsc_obj', 'data')]
#               )
# # @cache.memoize()
# def update_tabs(tab, result_obj, species, refs, stn_id, hsc_obj):

#     # print(flow_stn_id)

#     info_str = """
#             ### Intro
#             This is the [Environment Southland](https://www.es.govt.nz/) Freshwater Habitat Suitability dashboard.

#             ### Brief Guide
#             #### Selecting Species and Sites
#             A site can be selected from the map or directly on the dropdown list labeled "Site name". One or species can be selected from the dropdown list labeled "Select species". Both species and a site must be selected. There is also an option to select only the active flow sites (those with data in the last year) and all flow sites.

#             #### Habitat Suitability
#             The **Habitat Suitability** plot lists the selected reference method for the species over time according to the available gauging data for a particular site. Water depth and velocity from gaugings were used as input to the habitat suitability curves to derive the habitat suitability index. Please see [Jowett, I.G.; Hayes, J.W.; Duncan, M.J. (2008)](https://www.jowettconsulting.co.nz/home/reports) for a more detailed description on habitat suitability methods. The habitat suitability curves used in this app can be found in [this google sheet](https://docs.google.com/spreadsheets/d/1HMCAiJeVj2n89G9CkCV3cgQh2zoHIB2L3t_4BCb3COY/edit?usp=sharing).
#         """
#     print(tab)

#     fig1 = info_str

#     if (stn_id is not None) and (species is not None) and (len(refs) > 0):
#         if tab == 'hs_tab':
#             hsc_dict = utils.decode_obj(hsc_obj)
#             results = utils.decode_obj(result_obj)

#             fig1 = utils.render_plot(results, hsc_dict, species, refs)

#     if isinstance(fig1, str):
#         return dcc.Markdown(fig1)
#     else:
#         fig = dcc.Graph(
#                 # id = 'plots',
#                 figure = fig1,
#                 config={"displaylogo": False, 'scrollZoom': True, 'showLink': False}
#                 )

#         return fig


# @app.callback(
#     Output('selected_data', 'figure'),
#     [Input('result_obj', 'data')]
#     )
# def update_results_plot(result_obj):
#     """

#     """
#     base_dict = dict(
#             data = [dict(x=0, y=0)],
#             layout = dict(
#                 title='Click on the map to select a station',
#                 paper_bgcolor = '#F4F4F8',
#                 plot_bgcolor = '#F4F4F8',
#                 height = 400
#                 )
#             )

#     if len(result_obj) > 1:
#         results = utils.decode_obj(result_obj)
#         vars1 = list(results.variables)
#         parameter = [v for v in vars1 if 'dataset_id' in results[v].attrs][0]

#         results1 = results.isel(height=0, drop=True)

#         fig = go.Figure()

#         if 'geometry' in results.dims:
#             grps = results1.groupby('geometry')

#             for geo, grp in grps:
#                 if 'name' in grp:
#                     name = str(grp['name'].values)
#                     showlegend = True
#                 elif 'ref' in grp:
#                     name = str(grp['ref'].values)
#                     showlegend = True
#                 else:
#                     name = None
#                     showlegend = False

#                 times = pd.to_datetime(grp['time'].values)

#                 fig.add_trace(go.Scattergl(
#                     x=times,
#                     y=grp[parameter].values,
#                     showlegend=showlegend,
#                     name=name,
#         #                line={'color': col3[s]},
#                     opacity=0.8))
#         else:
#             results2 = results1[parameter].isel(lat=0, lon=0, drop=True)

#             times = pd.to_datetime(results2['time'].values)

#             fig.add_trace(go.Scattergl(
#                 x=times,
#                 y=results2.values,
#                 showlegend=False,
#                 # name=name,
#     #                line={'color': col3[s]},
#                 opacity=0.8))

#         # to_date = times.max()
#         # from_date = to_date - pd.DateOffset(months=6)

#         layout = dict(paper_bgcolor = '#F4F4F8', plot_bgcolor = '#F4F4F8', showlegend=True, height=780, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(l=20, r=20, t=20, b=20))

#         fig.update_layout(**layout)
#         fig.update_xaxes(
#             type='date',
#             # range=[from_date.date(), to_date.date()],
#             # rangeslider=dict(visible=True),
#             # rangeslider_range=[from_date, to_date],
#             # rangeslider_visible=True,
#             rangeselector=dict(
#                 buttons=list([
#                     dict(step="all", label='1y'),
#                     # dict(count=1, label="1 year", step="year", stepmode="backward"),
#                     dict(count=6, label="6m", step="month", stepmode="backward"),
#                     dict(count=1, label="1m", step="month", stepmode="backward"),
#                     dict(count=7, label="7d", step="day", stepmode="backward")
#                     ])
#                 )
#             )

#         fig.update_yaxes(autorange = True, fixedrange= False)

#     else:
#         fig = base_dict

#     return fig
