"""  
Suite of tools to be used in conjunction with the arcDemGetter/GenerateDerivatives scripts. Mainly a wrapper
for the tools found in dem_getter, made to include specific arc functionality.
"""

import urllib
import os
import arcpy
import sys
sys.path.append('..')
from dem_getter import dem_getter as dg

### GLOBAL VARIABLES

QUAD_URL='https://index.nationalmap.gov/arcgis/rest/services/USTopoAvailability/MapServer/0/query?f=json&where=CELL_NAME%3D%27{}%27+and+PRIMARY_STATE%3D%27{}%27&geometryType=esriGeometryEnvelope'

###

def arc_build_derivatives(inFiles:list, derivatives:list, filepath):
    """Calculates and saves derivatives for input files.
    Each derivative is given a name based on the original file name and a suffix 
    corresponding to that derivative.

    Args:
        inFiles (list): List of files to build derivatives for
        derivatives (list): Derivative(s) to compute for the input files
        filepath (str,list): Filepath to save outputs to, or list of filepaths of
        the same length as list of inFiles

    Returns:
        allDerivs (list): List of paths to the generated derivative files
    """

    arcpy.ResetProgressor()
    arcpy.SetProgressorLabel("Calculating derivatives")

    #If filepath isn't a list, make it a list
    if not isinstance(filepath,list):
        filepath = [filepath for i in inFiles]

    #Store a list of all the files that were generated
    allDerivs = []
    for file,filepath in zip(inFiles,filepath):
        arcpy.AddMessage('Derivatives for file: {}'.format(file))

        derivList = dg.compute_derivatives(file,derivatives,filepath)

        allDerivs.extend(derivList)

    return allDerivs

#%%

def arc_batch_download(inFiles, filePath):
    """Download from TNM the data from a list of paths, while informing the user via ArcGIS GUI
    of the size of their download

    Args:
        inFiles (list): List of web-hosted datasets
        filePath (_type_): Path to save the downloaded data to

    Returns:
        dlPaths (list): Paths to the downloaded data
    """

    #get file size of download
    size=0
    badRequests=0 #some urllib requests return a 404 error--skip these
    for lst in inFiles:
        try:
            req=urllib.request.Request(lst, method='HEAD')
            f = urllib.request.urlopen(req)
            size+=int(f.headers['Content-Length'])
    
            #Make the size of the download nicely legible
            sizeConvs = ['B','kB','MB','GB','TB'] #list of conversions
            pos = 0
            while (size > 1e3) & (pos < (len(sizeConvs)-1)):
                size/=1e3
                pos+=1
        except:
            badRequests+=1

    #notify user
    fileCount=len(inFiles)-badRequests
    warning_string = 'Downloading {} files; {:.2} {} of data.'.format(fileCount,size, sizeConvs[pos])
    arcpy.SetProgressor(type='step',message=warning_string,
    min_range=0,max_range=fileCount,step_value=1)

    #make directory to save files to if it doesn't already exist
    if not(os.path.isdir(filePath)):
        os.mkdir(filePath)

    #save paths to the downloaded data
    dlPaths=[]

    #download data
    for line in inFiles:
        #Strip off any whitespace
        #Get the 'name' of this file as the end of the filepath, we'll use this to save the file
        line = line.strip()
        name = line.split('/')[-1]

        #Keep track of progress
        downloadPath = os.path.join(filePath,name)
        try:
            urllib.request.urlretrieve(line,downloadPath)
            dlPaths.append(downloadPath)
            arcpy.SetProgressorPosition()
        except:
            arcpy.AddWarning("A product ({}) inside your search query is not available".format(name))
    
    return dlPaths

#%%

def format_for_quad_merge(quadName, savePath, mergeExt, inFile):
    """Build name for the merged quad data and get the EPSG of the input data

    Args:
        quadName (str): Name of the quad
        savePath (str): Folder to save the final product to
        mergeExt (str): The extension to save the final product as
        inFile (str): Path to one of the datasets to be merged

    Returns:
        outFile (str): Full file name to save the merged data as
        inEPSG (int): The EPSG code of the input file
    """

    outFile = dg.os.path.join(savePath,quadName.replace(" ", "")+'_{}'.format(mergeExt))

    inRas=arcpy.Raster(inFile)
    spatRef= inRas.spatialReference
    inEPSG=spatRef.factoryCode

    del inRas

    return outFile, inEPSG

#%%

def get_quad_extent(quadName, stateName, inEPSG):
    """Get the spatial boundary of the specified quad

    Args:
        quadName (str): Name of the quad
        stateName (str): State where the quad is located
        inEPSG (int): EPSG used to project the spatial boundary 

    Returns:
        extent (tuple): set of min/max coordinate pairs
    """

    x,y = dg._get_24kQuad_geom(quadName,stateName,inEPSG)

    arcpy.AddMessage((x,y))

    extent=([min(x),max(x)],[min(y),max(y)])

    return extent

#%%

def format_add_raster_data(rasterPaths, doStats=True, doPyramids=True):
    """Add raster data to the current map in ArcGIS

    Args:
        rasterPaths (list): Paths to the data to be added
        doStats (bool, optional): Calculate statistics for the data. Defaults to True.
        doPyramids (bool, optional): Calculate pyramids for the data. Defaults to True.
    """

    if not isinstance(rasterPaths,list):
        rasterPaths = [rasterPaths] #Force convert this to a list

    for rasterPath in rasterPaths:
        #load in raster
        ras=arcpy.Raster(rasterPath)
        if doStats:
           # arcpy.SetProgressorLabel("Calculating statistics")
            arcpy.CalculateStatistics_management(ras)
        if doPyramids:
           # arcpy.SetProgressorLabel("Building pyramids")
            arcpy.BuildPyramids_management(ras)

        #Add the data to the map
        prj=arcpy.mp.ArcGISProject('Current')
        m=prj.activeMap
        m.addDataFromPath(ras)

    return True

