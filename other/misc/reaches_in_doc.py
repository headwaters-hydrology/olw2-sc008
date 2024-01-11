#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:41:35 2023

@author: mike
"""
import os
from gistools import vector, rec
import geopandas as gpd
import pandas as pd
import numpy as np
from copy import copy
import nzrec
import booklet
import orjson
import geobuf
import shapely
from shapely.geometry import Point, Polygon, box, LineString, mapping

pd.options.display.max_columns = 10


#############################################
### Parameters

rec_rivers_gpkg = '/media/nvme1/data/NIWA/REC25_rivers/REC25_rivers_order3.gpkg'
doc_shp = '/media/nvme1/data/OLW/web_app/land_use/DOC_Public_Conservation_Land.shp'

doc_rec_gpkg = '/media/nvme1/data/OLW/doc_rec_3rd_and_above.gpkg'
doc_rec_csv = '/media/nvme1/data/OLW/doc_rec_3rd_and_above.csv'



############################################
### processing

rec0 = gpd.read_file(rec_rivers_gpkg)
doc0 = gpd.read_file(doc_shp)
doc0['geometry'] = doc0.buffer(0).simplify(1)

doc1 = doc0.unary_union

rec1 = rec0.iloc[rec0.sindex.query(doc1, predicate='intersects')]

rec1.to_file(doc_rec_gpkg)

pd.DataFrame(rec1.drop('geometry', axis=1)).to_csv(doc_rec_csv, index=False)







































