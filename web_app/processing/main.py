#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:18:20 2022

@author: mike
"""
import os
import pathlib
import utils

from land_cover_processing import land_cover_processing
from land_cover_reductions import land_cover_reductions

from rivers_delineate import rec_delin
# from rivers_reach_mappings import reach_mapping
# from rivers_catch import catch_agg
# from rivers_make_app_assets import process_assets
from rivers_land_cover_assignment import rivers_land_cover
from rivers_assign_errors import process_errors
from rivers_assign_flow import process_flows_rec
# from rivers_assign_loads import process_loads_rec
from rivers_monitoring_sites import rivers_monitoring_sites_processing

from lakes_geo_processing import lakes_geo_process
from lakes_delineation import lakes_catch_delin
from lakes_land_cover_extraction import lakes_lc_process
from lakes_conc_error import lakes_conc_error_processing

from gw_geo_processing import gw_geo_process
from gw_assign_errors import gw_process_errors_points

#####################################################
### Land use/cover

## process and clean up geo layers
land_cover_processing()

## Assign reductions
land_cover_reductions()

### Rivers

## Monitoring sites
rivers_monitoring_sites_processing()

## REC delineate all catchments that start at the sea and have a greater than 2 stream order
rec_delin()

## Land cover
rivers_land_cover()

## River flows
process_flows_rec()
# process_loads_rec()

## River error/power sims

# rivers_sims.py should be run via the terminal

process_errors()


###################################################
### Lakes

## Lakes locations
lakes_geo_process()

## Lakes catch delineation
lakes_catch_delin()

## Lakes land cover processing
lakes_lc_process()

## Lakes error/power sims

# lakes_sims.py should be run via the terminal

lakes_conc_error_processing()


###################################################
### GW

## GW locations
gw_geo_process()

## GW error/power sims

# gw_sims.py should be run via the terminal

gw_process_errors_points()







































