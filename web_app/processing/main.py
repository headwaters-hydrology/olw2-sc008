#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:18:20 2022

@author: mike
"""
import os
import pathlib

import sys
if '..' not in sys.path:
    sys.path.append('..')

import utils

from land_cover.lcdb_processing import lcdb_processing
from land_cover.land_cover_extra_layers import process_extra_geo_layers
from land_cover.land_cover_reductions import land_cover_reductions

from rivers.rivers_delineate import rec_delin
from rivers.rivers_land_cover_assignment import rivers_land_cover
from rivers.rivers_process_loads import process_loads
from rivers.rivers_monitoring_sites import rivers_monitoring_sites_processing
from rivers.rivers_assign_power_monitored import rivers_process_power_monitored
from rivers.rivers_assign_power_modelled import rivers_process_power_modelled
from rivers.rivers_catch_names import rivers_assign_catch_names
from rivers.rivers_marae import rivers_marae_processing

from ecology.eco_assign_power_modelled import eco_process_power_modelled
from ecology.eco_assign_power_monitored import eco_process_power_monitored
from ecology.eco_assign_reach_weights import eco_calc_river_reach_weights
from ecology.eco_monitoring_sites import eco_monitoring_sites_processing
from ecology.eco_catchment_stdev import eco_catchment_stdev_processing

from lakes.lakes_stdev_monitored import lakes_sd_conc
from lakes.lakes_geo_processing import lakes_geo_process
from lakes.lakes_delineation import lakes_catch_delin
from lakes.lakes_land_cover_assignment import lakes_land_cover
from lakes.lakes_process_loads import process_loads_lakes
from lakes.lakes_power_all import lakes_power_combo_processing

from gw.gw_geo_processing import gw_geo_process
from gw.gw_assign_power_monitored import gw_process_power_monitored

from high_loads.hl_process_loads import hl_process_loads

#####################################################
### Land use/cover

## Process and clean LCDB
lcdb_processing()

## Combine LCDB with other layers
process_extra_geo_layers()

## Assign reductions
land_cover_reductions()

####################################################
### Rivers

## REC delineate all catchments that start at the sea and have a greater than 2 stream order
rec_delin()

## Monitoring sites
rivers_monitoring_sites_processing()

## Assign catch names
rivers_assign_catch_names()

## process marae locations
rivers_marae_processing()

## Land cover
rivers_land_cover()

## River loads by indicator
process_loads()

## Route the reductions based on the default reductions

# rivers_route_reductions.py should be run via the terminal

## River error/power sims

# rivers_sims.py should be run via the terminal

## Process the powers associated with the other parameters
rivers_process_power_monitored()
rivers_process_power_modelled()

####################################################
### Ecology

## Monitoring sites
eco_monitoring_sites_processing()

## Catchment scale stdev
eco_catchment_stdev_processing()

## Reach weights
eco_calc_river_reach_weights()

## Sims for the catchment scale power calcs

# eco_sims_catch.py should be run via the terminal

## River error/power sims

# eco_sims.py should be run via the terminal

## Process the powers associated with the other parameters
eco_process_power_monitored()
eco_process_power_modelled()

###################################################
### Lakes

## Lakes stdev conc processing for modelling
lakes_sd_conc()

## Lakes locations
lakes_geo_process()

## Lakes catch delineation
lakes_catch_delin()

## Lakes land cover processing
lakes_land_cover()

## Lakes/rivers loads
process_loads_lakes()

## Lakes reductions routing

# lakes_route_reductions.py should be run via the terminal

## Lakes error/power sims

# lakes_sims.py should be run via the terminal

## Process the powers associated with the other parameters
lakes_power_combo_processing()


###################################################
### GW

## GW locations
gw_geo_process()

## GW error/power sims

# gw_sims.py should be run via the terminal

## Process the powers associated with the other parameters
gw_process_power_monitored()


#################################################
### High flow loads

## Process the reaches and sites results
hl_process_loads()



































