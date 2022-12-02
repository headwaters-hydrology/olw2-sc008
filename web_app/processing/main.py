#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:18:20 2022

@author: mike
"""
import os
import pathlib
import utils
from rec_rivers_delineate import rec_delin
from rec_reach_mappings import reach_mapping
from rec_catch import catch_agg
from make_app_assets import process_assets

#####################################################
### Run processing sequence

## REC delineate all catchments that start at the sea and have a greater than 2 stream order
rec_delin()

## Determine the upstream reaches of every reach - Takes 30 minutes to run!
reach_mapping()

## Process catchments
catch_agg()

## Process data for web app assets
process_assets()










































