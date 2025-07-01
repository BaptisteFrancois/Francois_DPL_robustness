
import tempfile
import geopandas as gpd
import sys
import os
import importlib.util


#from dem_getter.dem_getter import dem_getter # required to import process_watershed_flow_length
from GIS_watershed_tools.extraction_watershed_features_from_DEM import process_watershed_flow_length


# WHitebox working directory (looks like the absolute path is required)
#wbt_work_dir = 'D:/17_TOVA/DPL_robustness/temporary_working_dir'

# Load the shapefile of the 671 basins
basin_shapefile_file = '../data/shapefiles/CAMELS/HCDN_nhru_final_671.shp'
# Read the shapefile using geopandas
basin_shapefile = gpd.read_file(basin_shapefile_file)
# Convert crs to 4326 (WSG84); required for dem_getter
basin_shapefile = basin_shapefile.to_crs(epsg=4326)


with tempfile.TemporaryDirectory() as wbt_work_dir:
    print(f"Using temporary working directory: {wbt_work_dir}")  
    # Process the watershed flow length for the basins
    flow_length_df = process_watershed_flow_length(
        basin_shapefile=basin_shapefile,
        wbt_work_dir=wbt_work_dir,
        keep_dem_raster=False
    )
