#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:55:36 2021

@author: mike
"""
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pathlib

##############################################
### The app

app_base_path = pathlib.Path('/assets')

def sidebar():
    return html.Div(
        dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.Div('- ' + page["name"], className="ms-2"),
                    ],
                    href=page["path"],
                    active="exact",
                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
            className="bg-light",
        )
    )


app = dash.Dash(__name__, use_pages=True)
server = app.server


app.layout = html.Div(children=[
    html.Div([dcc.Link(html.Img(src=str(app_base_path.joinpath('our-land-and-water-logo.svg'))), href='https://ourlandandwater.nz/'),
              html.H3('Contents'),
              sidebar(),
              ], className='one column', style={'margin': 0, 'margin-top': 15}),
    html.Div([dash.page_container
              ], className='eleven columns', style={'margin': 0})
    ])





# app.layout = dbc.Row(
#         [dbc.Col(sidebar(), width=2), dbc.Col(dash.page_container, width=10)]
#     )


# html.Div([
# 	html.H3('Multi-page app with Dash Pages'),

#     html.Div(
#         [
#             html.Div(
#                 dcc.Link(
#                     f"{page['name']} - {page['path']}", href=page["relative_path"]
#                 )
#             )
#             for page in dash.page_registry.values()
#         ]
#     ),

# 	dash.page_container
# ])
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=80)

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8000)
