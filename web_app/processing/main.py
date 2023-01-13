#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:18:20 2022

@author: mike
"""
import os
import pathlib
import utils
from rivers_delineate import rec_delin
# from rivers_reach_mappings import reach_mapping
from rivers_catch import catch_agg
from rivers_make_app_assets import process_assets
from rivers_land_cover_extraction import rivers_land_cover
from rivers_assign_conc import process_conc

from lakes_locations import lakes_location_process
from lakes_rec_delineation import lakes_catch_delin
from lakes_land_cover_extraction import lakes_lc_process
from lakes_conc_error import lakes_conc_error_processing


#####################################################
### Rivers

## REC delineate all catchments that start at the sea and have a greater than 2 stream order
rec_delin()

## Determine the upstream reaches of every reach - Takes 30 minutes to run!

# rivers_reach_mappings.py should be run from the terminal

## Process catchments
catch_agg()

## Land cover
rivers_land_cover()

# rivers_land_cover_clean.py should be run via the terminal

## River error/conc sims
process_conc()

# rivers_sims.py should be run via the terminal

## Extra assets for web app
process_assets()

###################################################
### Lakes

## Lakes location
lakes_location_process()

## Lakes catch delineation
lakes_catch_delin()

## Lakes land cover processing
lakes_lc_process()

## Lakes error/conc sims
lakes_conc_error_processing()

# lakes_sims.py should be run via the terminal


###################################################
### GW











































