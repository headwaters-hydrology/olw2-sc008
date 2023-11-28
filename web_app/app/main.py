#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:55:36 2021

@author: mike
"""
import dash
from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
# from dash_iconify import DashIconify
import pathlib
from time import sleep
import hdf5plugin

##############################################
### The app
# sleep(10)
app = dash.Dash(__name__,
                use_pages=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP]
                )
server = app.server

app_base_path = pathlib.Path('/assets')

page_path_names = {v['path']: v['description'] for k, v in dash.page_registry.items()}

def create_nav_link(label, href):
    return dcc.Link(
        dmc.Group(
            [
                # dmc.ThemeIcon(
                #     DashIconify(icon=icon, width=18),
                #     size=30,
                #     radius=30,
                #     variant="light",
                # ),
                dmc.Text(label, size="sm", color="grey"),
            ]
        ),
        href=href,
        style={"textDecoration": "none"},
    )


def create_sidebar_children(pages):
    """

    """
    list1 = [
        dmc.Group(
        # direction="column",
        children=[
            create_nav_link(
                label=dmc.Text("Home", style={'font-size': 16}),
                href="/",
                ),
            ],
            ),
        dmc.Divider(
            label=dmc.Text("User Guides", style={'font-size': 16}), style={"marginBottom": 20, "marginTop": 20}
            ),
        dmc.Divider(
            label=dmc.Text("Rivers", style={'font-size': 16}), style={"marginBottom": 10, "marginTop": 20}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.rivers_wq"]['title'], href=pages["pages.rivers_wq"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.rivers_wq_sites"]['title'], href=pages["pages.rivers_wq_sites"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.rivers_eco"]['title'], href=pages["pages.rivers_eco"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.rivers_eco_sites"]['title'], href=pages["pages.rivers_eco_sites"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.rivers_hfl"]['title'], href=pages["pages.rivers_hfl"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        # dmc.Group(
        #     children=[
        #         create_nav_link(
        #             label=pages["pages.rivers_hfl_sites"]['title'], href=pages["pages.rivers_hfl_sites"]["path"]
        #             )
        #         ], style={"marginBottom": 10}
        #     ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.land_cover"]['title'], href=pages["pages.land_cover"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Divider(
            label=dmc.Text("Lakes", style={'font-size': 16}), style={"marginBottom": 10, "marginTop": 20}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.lakes_wq"]['title'], href=pages["pages.lakes_wq"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.lakes_wq_sites"]['title'], href=pages["pages.lakes_wq_sites"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        dmc.Divider(
            label=dmc.Text("Groundwater", style={'font-size': 16}), style={"marginBottom": 10, "marginTop": 20}
            ),
        dmc.Group(
            children=[
                create_nav_link(
                    label=pages["pages.gw_wq"]['title'], href=pages["pages.gw_wq"]["path"]
                    )
                ], style={"marginBottom": 10}
            ),
        ]

    return list1


# sidebar = dmc.Navbar(
#     fixed=True,
#     width={"base": 185},
#     position={"top": 80},
#     px=10,
#     children=[
#         dmc.ScrollArea(
#             offsetScrollbars=True,
#             type="scroll",
#             children=create_sidebar_children(dash.page_registry)
#             )
#         ]
#     )

app.layout = html.Div(
    [
      # dmc.Header(
      #    height=60,
      #    fixed=True,
      #    px=25,
      #    children=[
      #        dmc.Grid(
      #            # gutter='xl',
      #            style={"height": 60},
      #            children=[
      #                dmc.Col(
      #                    dmc.Anchor(
      #                        'Mitigation Effectiveness Monitoring Design',
      #                        size="xl",
      #                        href="/",
      #                        underline=False,
      #                        # align='center',
      #                        # style={'vertical-align': 'middle'}

      #                    ),
      #                    span=6,
      #                    style={'padding': '20px 0'}
      #                    # style={'vertical-align': 'middle'}
      #                    ),
      #                dmc.Col(
      #                    dmc.Text(
      #                        '',
      #                        id='title',
      #                        size=28,
      #                        ),
      #                    span=3,
      #                    style={'padding': '20px 0'}
      #                    ),
      #                dmc.Col(
      #                    dmc.Anchor(
      #                        dmc.Image(
      #                            src=str(app_base_path.joinpath('our-land-and-water-logo.svg')),
      #                            fit='cover',
      #                            width='90%'
      #                            ),
      #                        href='https://ourlandandwater.nz/'
      #                        ),
      #                    span=3,
      #                    offset=0
      #                    ),
      #                ]
      #            )
      #        ]
      #    ),

      #    sidebar,
        dmc.Container(
            dash.page_container,
            size="xl",
            # pt=20,
            style={"margin-top": 0, 'margin-left': 0, 'margin-right': 0, 'margin-bottom': 0},
            # style={"margin-top": 80, 'margin-left': 200, 'margin-right': 20},
        ),
    ],
)

# @callback(
#     Output('title', 'children'),
#     Input('_pages_location', 'pathname'),
#     prevent_initial_call=True
#     )
# def updated_title(path):
#     # print(path)
#     if isinstance(path, str):
#         title = page_path_names[path]
#         if title == 'Home':
#             # title = 'Mitigation Effectiveness Monitoring Design'
#             title = ''

#         return title
#     else:
#         raise dash.exceptions.PreventUpdate


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=80)

# if __name__ == '__main__':
#     app.run_server(debug=True, host='0.0.0.0', port=8000)
