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

### Colorbars
colorbar_base = dl.Colorbar(style={'opacity': 0})
base_reach_style = dict(weight=4, opacity=1, color='white')

## Power
indices = list(range(len(param.ctg) + 1))
colorbar_power = dl.Colorbar(min=0, max=len(param.ctg), classes=indices, colorscale=param.colorscale_power, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=param.ctg, width=300, height=30, position="bottomright")

## Land cover reductions
indices_reductions = list(range(len(param.ctg_reductions) + 1))
colorbar_reductions = dl.Colorbar(min=0, max=len(param.ctg_reductions), classes=indices_reductions, colorscale=param.colorscale_reductions, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=param.ctg_reductions, width=300, height=30, position="bottomright")

## Eco categories
ctg_weights = ['Low', 'Moderate', 'High']

indices = list(range(len(ctg_weights) + 1))
colorbar_weights = dl.Colorbar(min=0, max=len(ctg_weights), classes=indices, colorscale=param.colorscale_weights, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg_weights, width=300, height=30, position="bottomright")

## High flow load
indices = list(range(len(param.hfl_ctg) + 1))
colorbar_hfl = dl.Colorbar(min=0, max=len(param.ctg), classes=indices, colorscale=param.hfl_colorscale, tooltip=True, tickValues=[item + 0.5 for item in indices[:-1]], tickText=param.hfl_ctg, width=300, height=30, position="bottomright")




