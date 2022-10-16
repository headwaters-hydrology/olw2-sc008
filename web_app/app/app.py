#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:55:36 2021

@author: mike
"""
import dash


##############################################
### The app

app = dash.Dash(__name__,  url_base_pathname = '/', prevent_initial_callbacks=True)
server = app.server

