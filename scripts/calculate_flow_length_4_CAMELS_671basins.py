
# CONDA ENV: rivercloud_gis

# This script extracts the Livneh-Lusu dataset for the 671 basins in the CAMELS dataset.


import os
import sys
import io
import contextlib
# Create a dummy stream to capture stdout.
dummy_stream = io.StringIO()

# Import required libraries
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from whitebox.whitebox_tools import WhiteboxTools

sys.path.append('../dem_getter')
from dem_getter import dem_getter


# Local directory of the LOCAc2 data
livneh_lusu_dir = 'H:/Livneh_lusu_2020/'

# WHitebox working directory (looks like the absolute path is required)
wbt_work_dir = 'D:/17_TOVA/DPL_robustness/temporary_working_dir'

# Load the shapefile of the 671 basins
basin_shapefile_file = '../data/shapefiles/CAMELS/HCDN_nhru_final_671.shp'
# Read the shapefile using geopandas
basin_shapefile = gpd.read_file(basin_shapefile_file)
# Convert crs to 6933 to equal area projection for area calculations
basin_equal_area = basin_shapefile.to_crs(epsg=6933)
# Convert crs to 4326 (WSG84); required for dem_getter
basin_shapefile = basin_shapefile.to_crs(epsg=4326)

# List to store the deviation in area for each basin. This is the difference between the area of 
# the largest polygon in a MultiPolygon and the area of the basin as defined in the shapefile.
# When the `geometry` of the basin is a Polygon, the deviation should be zero (or very small because
# of rounding errors). When the `geometry` is a MultiPolygon, the deviation is the difference
# between the area of the largest polygon and the area of the basin as defined in the shapefile.
deviation_area = []  

# Loop over the basins
for index, row in basin_shapefile.iterrows():
    basin_id = str(row['hru_id']).zfill(8)  # Ensure the basin ID is zero-padded to 8 digits

    # Get the bounding box of the selected basin
    xMin, yMin, xMax, yMax = basin_shapefile.iloc[index].geometry.bounds

    # Use dem_getter to download and clip the DEM for the basin defined by the bounding box 'bounds'
    paths = dem_getter.get_aws_paths(
        dataset='NED_1as',
        xMin=xMin,
        yMin=yMin,
        xMax=xMax,
        yMax=yMax,
        filePath=None,
        inputEPSG=4326,
        doExcludeRedundantData=True
    )

    dem_getter.batch_download(
        dlList=paths,
        folderName='../temporary_working_dir',
        doForceDownload=True
    )

    # Merge the downloaded DEM files into a single xarray dataset
    dem_files = [os.path.join('../temporary_working_dir', f) for f in 
                 os.listdir('../temporary_working_dir') if f.endswith('.tif')]
    
    src_files = [rasterio.open(f) for f in dem_files]
    
    # Plot the dem files to check if they are loaded correctly
    #fig, axes = plt.subplots(1,2,figsize=(10, 10))
    #for src, ax in zip(src_files, axes.ravel()):
    #    # Read the DEM data and plot it
    #    ax.imshow(src.read(1),
    #                cmap='terrain',
    #                extent=(src.meta['transform'][2],
    #                        src.meta['transform'][2] + src.meta['transform'][0] * src.width,
    #                        src.meta['transform'][5] + src.meta['transform'][4] * src.height,
    #                        src.meta['transform'][5]),
    #                aspect='auto')
    #plt.show()

    mosaic, out_trans = merge(src_files)
    
    # Plot the merged DEM to check if it looks correct
    #fig, ax = plt.subplots(figsize=(10, 10))
    #plt.imshow(mosaic[0], cmap='terrain', 
    #           extent=(src.meta['transform'][2],
    #                        out_trans[2] + out_trans[0] * mosaic.shape[2],
    #                        out_trans[5] + out_trans[4] * mosaic.shape[1],
    #                        out_trans[5]),
    #           aspect='auto')
    #gpd.GeoSeries(basin_shapefile.iloc[index].geometry.boundary).plot(ax=ax, color='red', linewidth=1)
    #plt.colorbar(label='Elevation (m)')
    #plt.show()


    # Update the metadata from one of the source DEMs
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_trans
    })

    # Write the merged DEM to a new file
    out_merged_image = f'../temporary_working_dir/dem_{basin_id}.tif'
    with rasterio.open(out_merged_image, 'w', **out_meta) as dest:
        dest.write(mosaic)

    # Close the source files
    for src in src_files:
        src.close()

    # Delete the temporary downloaded DEM files
    for f in dem_files:
        os.remove(f)

    # Read the merged DEM file and clip it to the basin geometry
    with rasterio.open(out_merged_image) as src:
        # Project the geometry to the DEM's CRS
        projected_basin = basin_shapefile.copy()
        projected_basin = projected_basin.to_crs(src.crs)
        # Mask the DEM with the basin geometry
        out_image, out_transform = mask(src, [projected_basin.iloc[index].geometry], crop=True)

        # Project the clipped DEM to the WGS84 coordinate system (meters) (EPSG:6933)
        # This is necessary for the WhiteboxTools operations that follow.

        out_meta = src.meta.copy()

    # Update the metadata for the masked DEM
    out_meta.update({
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform
    })

    # Plot the clipped DEM to check if it looks correct
    out_image_without_nodata = np.where(out_image == src.nodata, np.nan, out_image)
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(out_image_without_nodata[0], cmap='terrain',
                extent=(out_transform[2],
                          out_transform[2] + out_transform[0] * out_image.shape[2],
                          out_transform[5] + out_transform[4] * out_image.shape[1],
                          out_transform[5]),
                aspect='auto')
    gpd.GeoSeries(projected_basin.iloc[index].geometry.boundary).plot(ax=ax, color='red', linewidth=1)
    plt.colorbar(label='Elevation (m)')
    plt.title(f'Clipped DEM for Basin {basin_id}')
    plt.show()

    
    clipped_dem_file = f'../temporary_working_dir/dem_{basin_id}_clipped.tif'

    # Write the clipped DEM to a new file
    with rasterio.open(clipped_dem_file, 'w', **out_meta) as dest:
        dest.write(out_image)

    # Delete the original merged DEM file
    os.remove(out_merged_image)

    # Reproject the clipped DEM to the WGS84 coordinate system (EPSG:6933)
    # This is necessary for the WhiteboxTools operations that follow.
    with rasterio.open(clipped_dem_file) as src:
        # Calculate the transform and metadata for the reprojected DEM
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:6933', src.width, src.height, *src.bounds)
        
        # Update the metadata for the reprojected DEM
        # Note: 'EPSG:6933' is the WGS84 coordinate system in meters, suitable for distance calculations.
        out_meta = src.meta.copy()
        out_meta.update({
            'crs': 'EPSG:6933',
            'transform': transform,
            'width': width,
            'height': height
        })

        # Create a new file for the reprojected DEM
        reprojected_dem_file = f'../temporary_working_dir/dem_{basin_id}_clipped_reprojected.tif'
        
        with rasterio.open(reprojected_dem_file, 'w', **out_meta) as dest:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dest, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:6933',
                    resampling=Resampling.nearest
                    )

    # Plot the reprojected DEM to check if it looks correct
    #with rasterio.open(reprojected_dem_file) as src: 
    #    out_image = src.read(1)
    #    out_image_without_nodata = np.where(out_image == src.nodata, np.nan, out_image)
    #    fig, ax = plt.subplots(figsize=(10, 10))
    #    plt.imshow(out_image_without_nodata, cmap='terrain',
    #                extent=(src.transform[2],
    #                        src.transform[2] + src.transform[0] * src.width,
    #                        src.transform[5] + src.transform[4] * src.height,
    #                        src.transform[5]),
    #                aspect='auto')
    #    gpd.GeoSeries(basin_equal_area.iloc[index].geometry.boundary).plot(ax=ax, color='red', linewidth=1)
    #    plt.colorbar(label='Elevation (m)')
    #    plt.title(f'Reprojected Clipped DEM for Basin {basin_id}')
    #    plt.show()
    
    

    # Set up the WhiteboxTools instance and set the working directory
    wbt = WhiteboxTools()
    wbt.set_working_dir(wbt_work_dir)

    with contextlib.redirect_stdout(dummy_stream):
        
        # Fill the depressions in the DEM to ensure that there are no sinks
        wbt.fill_depressions(
            dem=f'dem_{basin_id}_clipped_reprojected.tif',
            output=f'dem_{basin_id}_filled.tif',
            fix_flats=True,
        #    flat_increment=0.01
        )
        
        # Compute the D8 flow direction raster
        wbt.d8_pointer(
            dem=f'dem_{basin_id}_clipped_reprojected.tif',
            #dem=f'dem_{basin_id}_filled.tif',
            output=f'flow_direction_{basin_id}.tif'
        )

        # Compute the flow length. This is the length of the flow path from each cell to the outlet.
        wbt.max_upslope_flowpath_length(
            f'flow_direction_{basin_id}.tif',
            f'flow_length_{basin_id}.tif'
        )

        # Convert the basin shapefile to a raster mask
        
        # Save the basin mask as a .shp file
        gpd.GeoDataFrame(
            {'hru_id': [int(basin_id)]},
            geometry=[projected_basin.iloc[index].geometry],
            crs=projected_basin.crs
        ).to_file(f'{wbt_work_dir}/basin_mask_{basin_id}.shp')



        wbt.vector_polygons_to_raster(
            i=f'basin_mask_{basin_id}.shp',
            output=f'basin_mask_{basin_id}.tif',
            field='hru_id',
        )

        wbt.longest_flowpath(
            dem=f'dem_{basin_id}_clipped_reprojected.tif',
            basins=f'basin_mask_{basin_id}.tif',
            output=f'longest_flowpath_{basin_id}.shp'
        )
        


    # Plot the flow direction raster to check if it looks correct
    with rasterio.open(f'../temporary_working_dir/flow_direction_{basin_id}.tif') as src:
        flow_direction = src.read(1)
        flow_direction_without_nodata = np.where(flow_direction == src.nodata, np.nan, flow_direction)
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(flow_direction_without_nodata, cmap='terrain',
                   extent=(src.transform[2],
                           src.transform[2] + src.transform[0] * src.width,
                           src.transform[5] + src.transform[4] * src.height,
                           src.transform[5]),
                   aspect='auto')
        gpd.GeoSeries(basin_equal_area.iloc[index].geometry.boundary).plot(ax=ax, color='red', linewidth=1)
        plt.colorbar(label='Flow Direction')
        plt.title(f'Flow Direction for Basin {basin_id}')
        plt.show()

    # Plot the flow length raster to check if it looks correct
    with rasterio.open(f'../temporary_working_dir/flow_length_{basin_id}.tif') as src:
        flow_length = src.read(1)
        flow_length_without_nodata = np.where(flow_length == src.nodata, np.nan, flow_length)  # Replace nodata with NaN
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(flow_length_without_nodata, cmap='terrain',
                    extent=(src.transform[2],
                            src.transform[2] + src.transform[0] * src.width,
                            src.transform[5] + src.transform[4] * src.height,
                            src.transform[5]),
                    aspect='auto')
        gpd.GeoSeries(basin_equal_area.iloc[index].geometry.boundary).plot(ax=ax, color='red', linewidth=1)
        plt.colorbar(label='Flow Length (m)')
        plt.title(f'Flow Length for Basin {basin_id}')
        plt.show()
    
    
    # Extract the maximum flow length
    with rasterio.open(f'../temporary_working_dir/flow_length_{basin_id}.tif') as src:
        flow_length = src.read(1)
        max_flow_length = np.nanmax(flow_length)

    sdf



#    # Check if the basin is a MultiPolygon or a Polygon
#    if basin_shapefile.iloc[index].geometry.type == 'MultiPolygon':
#        # Find the index of the largest polygon in the MultiPolygon
#        index_largest_area = \
#            np.argmax(np.array([g.area for g in basin_equal_area.iloc[index].geometry.geoms]))
#        selected_area = basin_equal_area.iloc[index].geometry.geoms[index_largest_area].area
#        geom = basin_equal_area.iloc[index].geometry.geoms[index_largest_area]
#    else:
#        geom = basin_shapefile.iloc[index].geometry
#        selected_area = basin_shapefile.iloc[index].geometry.area

#selected_basin_for_area_calculation = basin_equal_area.iloc[index]

# Get the geometry of the selected basin
# This code snippet is required because `dem_getter` requires 
#if selected_basin_for_area_calculation.geometry.type == 'MultiPolygon':
#    # Find the index of the largest polygon in the MultiPolygon
#    index_largest_area = \
#        np.argmax(np.array([g.area for g in selected_basin_for_area_calculation.geometry.geoms]))
#    selected_area = selected_basin_for_area_calculation.geometry.geoms[index_largest_area].area
#    geom = selected_basin.geometry.geoms[index_largest_area]
#else:
#    geom = selected_basin.geometry
#    selected_area = selected_basin_for_area_calculation.geometry.area

# Calculate the error made by selecting the largest polygon (km2)
#deviation_area.append((selected_area - selected_basin_for_area_calculation.AREA) / 1e6 )

# create a geodataframe with the selected basin only
#selected_basin_gdf = gpd.GeoDataFrame(index=[index], geometry=[geom], crs=4326)

