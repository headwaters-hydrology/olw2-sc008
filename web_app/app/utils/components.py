#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:45:11 2023

@author: mike
"""
import dash_leaflet as dl
import utils.parameters as param

##################################################
#### Global components

### Colorbar
colorbar_base = dl.Colorbar(style={'opacity': 0})
base_reach_style = dict(weight=4, opacity=1, color='white')

indices = list(range(len(param.ctg) + 1))
colorbar_power = dl.Colorbar(min=0, max=len(param.ctg), classes=indices, colorscale=param.colorscale_power, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=param.ctg, width=300, height=30, position="bottomright")

