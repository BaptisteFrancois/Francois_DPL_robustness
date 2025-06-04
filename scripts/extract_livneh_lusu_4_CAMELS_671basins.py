
# CONDA ENV: rivercloud_gis

# This script extracts the Livneh-Lusu dataset for the 671 basins in the CAMELS dataset.

import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd




# Local directory of the LOCAc2 data
livneh_lusu_dir = 'H:/Livneh_lusu_2020/'

# Load the shapefile of the 671 basins
basin_shapefile_file = '../data/shapefiles/CAMELS/HCDN_nhru_final_671.shp'
basin_shapefile = gpd.read_file(basin_shapefile_file)


