# CONDA ENV: tova_crp

# This script extracts the Livneh-Lusu dataset for the 671 basins in the CAMELS dataset.

# Import required libraries
import numpy as np
import geopandas as gpd
from ClimProjTools.process_LivnehLusu import extract_livneh_lusu_data
import multiprocessing as mp

# Local directory of the LOCAc2 data
livneh_lusu_dir = 'H:/Livneh_lusu_2020/'
output_dir = '../data/livneh_lusu/'
tmp_work_dir = '../temporary_working_dir'

# Load the shapefile of the 671 basins
basin_shapefile_file = '../data/shapefiles/CAMELS/HCDN_nhru_final_671.shp'
# Read the shapefile using geopandas
basin_shapefile = gpd.read_file(basin_shapefile_file)

# Convert crs to 4326 (WSG84); required for dem_getter
basin_shapefile = basin_shapefile.to_crs(epsg=4326)

# Split the basin shapefile into chunks for parallel processing
num_processes = 4
basin_chunks = np.array_split(basin_shapefile, num_processes)

params = [
    (
        basin_chunk,  # your specific basin (or list of basins) for this job
        (1950, 2018),  # the years range
        livneh_lusu_dir, # the directory where the Livneh-Lusu data is stored
        output_dir, # the directory where the output files will be saved
        'livneh_lusu_2020_temp_and_wind.2021-05-02.{}.nc', # Livneh-Lusu temperature and wind file template
        'livneh_unsplit_precip.2021-05-02.{}.nc', # Livneh-Lusu precipitation file template
        False,  # Overwrite flag
        tmp_work_dir, # Temporary working directory
        True    # Plot grid cells flag
    )
    for basin_chunk in basin_chunks
]

if __name__ == '__main__':

    # extract the Livneh-Lusu data for each chunk in parallel
    mp.Pool(num_processes).starmap(extract_livneh_lusu_data, params)