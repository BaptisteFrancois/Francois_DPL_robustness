
#%%
from osgeo import gdal
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.ndimage as ndi
import os
import urllib

#Standard keyword arguments for optional NDI convolve calls,
#centers convolution on the center of the kernel and sets results
#the kernel overlaps the marging of the DEM to no data
NDICONVOLVE_KWARGS = {'mode' : 'constant','cval' : np.nan, 'origin' : 0}

#Dictionary that maps an extension to a gdal raster driver name
FILE_EXT_DICT={'asc': 'AAIGrid', 'tif': 'GTiff', 'img': 'HFA', 'xyz': 'XYZ',
               'jpg': 'JPEG', 'hdf':'HDF4', 'nc':'netCDF'}

def get_raster_as_grid(raster):
    """Loads in a digital elevation model as a numpy array and converts any no-data to np.nan

        Args:
            raster (str OR gdal.Dataset): The path to a single band, gdal readable, digital elevation model OR
                an already loaded raster dataset

        Returns:
            rasterGrid (numpy.ndarray): The elevation data stored as a grid
            dx (float): The x coordinate spacing of the grid
            dy (float): The y coordinate spacing of the grid
            zFactor (float): The estimated tranformation between horizontal and vertical coordinates

        Raises:
            Exception: Input is neither a path to a raster or a gdal.Dataset
        """
    #If this is a raster dataset
    if isinstance(raster,gdal.Dataset):
        doClose = False
    
    #If this is a file
    elif os.path.isfile(raster):   
        #Get the raster grid
        raster = gdal.Open(raster)

        #Close this file after the operation completes
        doClose = True
    else:
        raise Exception('Specified raster is neither a path to a raster or a gdal.Dataset. Please specify a valid path.')

    rasterGrid = raster.ReadAsArray().astype(float)
    NDV = raster.GetRasterBand(1).GetNoDataValue()

    #Mask out NDVs as nan
    rasterGrid[rasterGrid==NDV] = np.nan

    #Estimate the zFactor for this dataset (if necessary)
    zFactor = estimate_z_factor(raster)

    # Grab the basic header information (xUL, dx, rot1, yUL, rot2, dy)
    geotransform = raster.GetGeoTransform()  
    
    dx = geotransform[1]
    dy = geotransform[-1]

    if doClose:
        raster = None #Close the raster

    return rasterGrid, dx, dy, zFactor

def plot_gdal_source(source:gdal.Dataset,ax:plt.Axes = None,
                     doTrimToPercentiles:bool = False,
                     cmap:str = 'Greys',
                     valueMinMaxPercentiles:list = [10,90],
                     doAddColorbar:bool = False,
                     colorbarKwargs:dict = {},
                     **kwargs):
    """ Create a plot using matplotlib.pyplot.imshow that depicts the raster values.

    The colorbarKwargs dictionary is used to adjust properties of the colorbar, if one was requested.

    Other keyword arguments are passed to plt.imshow to adjust the plot of this raster.

    Args:
        source (gdal.Dataset): A gdal.Dataset (e.g., loaded with source = gdal.Open())
            of a single band raster.
        ax (plt.Axes, optional): A matplotlib axis for calling axs.imshow on. Defaults to None.
        doTrimToPercentiles (bool, optional): Whether to trim the colorscale to
            a percentile of observed values. Defaults to False.
        cmap (str, optional): The matplotlib named colormap to plot the image on. Defaults to 'Greys'.
        valueMinMaxPercentiles (list, optional): Minimum and maximum percentiles of data to
            define the range of the colorbar legend. Only relevant of doTrimToPercentiles is True.
            Defaults to [10,90].
        doAddColorbar (bool, optional): Whether to add a colobar legend to the plot. Defaults to False.
        colorbarKwargs (dict, optional): Dictionary of keyword arguments to pass to matplotlib.pyplot.colorbar
        to adjust the colorbar if present.

    Returns:
        ax (matplotlib.pyplot axis): Either the one created for the plot or the one passed to the function.
    """
    if ax is None:
        f,ax = plt.subplots(1,1)

    #Read the array of data
    grid = source.ReadAsArray().astype(float)

    #Mask the no data values
    grid[grid==source.GetRasterBand(1).GetNoDataValue()]=np.nan
    # grid = np.ma.masked_values(grid,source.GetRasterBand(1).GetNoDataValue())

    #Stretch the min/max colormap values to percentiles
    if doTrimToPercentiles:
        vmin,vmax = np.nanpercentile(grid,valueMinMaxPercentiles)
        kwargs['vmin'] = vmin
        kwargs['vmax'] = vmax
    
    #If there aren't vmin or vmax values in the keyword arguments, add them
    #because they (might) be needed for the colorbar below
    if not('vmin' in kwargs):
        kwargs['vmin'] = np.nanmin(grid)
    if not('vmax' in kwargs):
        kwargs['vmax'] = np.nanmax(grid)

    #Get the information necessary to determine the grid coordinates
    nrows, ncols = grid.shape
    ulx, xres, _, uly, _, yres = source.GetGeoTransform()
    extent = [ulx, ulx+xres*ncols, uly+yres*nrows, uly]

    #Plot the grid
    ax.imshow(grid, extent=extent,cmap=cmap, **kwargs)

    #If requested, add the colorbar
    if doAddColorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim([vmin,vmax])

        plt.colorbar(sm,ax=ax, **colorbarKwargs)

    return ax

def duplicate_raster_with_array(array:np.ndarray, templateRaster,
                                newFileName:str, doReturnGdalSource:bool = False):
    """Create a copy of the specified raster with the data present in the numpy ndarray 'array',
    and save the file with a new suffix in the same directory as the source

    Args:
        array (numpy.ndarray): Array with the same dimensions as that specified in pathToRaster
        templateRaster (str,path OR gdal.Dataset): Path to a raster dataset that can be loaded by gdal, 
        OR a dataset that has already been loaded
        newFileName (str, path): New path to save the new raster dataset
        doReturnGdalSource (bool): Should the gdal.DataSource object be returned (True), or should
            the file be closed (False, default)

    Returns:
        outDataset (gdal.Dataset OR None): If doReturnGdalSource is True, returns the opened gdal.Datasource object for the new
        file. Otherwise returns None

    Raises:
        Exception: Input is neither a path to a raster or a gdal.Dataset
    """
    #If this is a raster dataset
    if isinstance(templateRaster,gdal.Dataset):
        doClose = False
    
    #If this is a file
    elif os.path.isfile(templateRaster):   
        #Get the raster grid
        templateRaster = gdal.Open(templateRaster)

        #Close this file after the operation completes
        doClose = True
    else:
        raise Exception('Specified raster is neither a path to a raster or a gdal.Dataset. Please specify a valid path.')

    #Hmmm.... would much rather get this dynamically from the raster, but for now just setting to 32 bit float
    gdal_datatype = gdal.GDT_Float32

    #What driver (e.g., file type) was used?
    if (not newFileName) or (newFileName.lower() == 'memory'):
        #If no path was specified, get something in memory
        drvr = gdal.GetDriverByName("MEM")
        doReturnGdalSource = True
    else:
        #If a path was specified, this is to be saved - get the filetype from the saved file
        drvr = templateRaster.GetDriver()

        if drvr.GetDescription() =='MEM': #if the user wants the raster saved but a memory file is input,
            #try to get the driver from the save path extension
            ext=os.path.splitext(os.path.split(newFileName)[-1])[-1].split('.')[-1]
            try:
                drvr=gdal.GetDriverByName(FILE_EXT_DICT[ext])

            except: #if the save path ext isn't in the dict then assign it to GTiff
                import warnings
                warnings.warn("The requested file format is not available. Saving as GTiff instead.")
                drvr=gdal.GetDriverByName("GTiff")

    #Get relevant info from the srcDataset
    NDV = templateRaster.GetRasterBand(1).GetNoDataValue() #Get the no data value

    #Copy the array so we can update any nan values to the NDV
    arrayCopy = array.copy()
    arrayCopy[np.isnan(arrayCopy)] = NDV

    #Create the output dataset
    outDataset = drvr.Create(newFileName,templateRaster.RasterXSize,
                             templateRaster.RasterYSize,1,gdal_datatype)
    
    #NOTE! Before I always set the NDV after writing an array... but in Arc, when using 'MEM' driver...
    #this was causing all values to be replaced with NDV
    outDataset.GetRasterBand(1).SetNoDataValue(NDV) #Write the no data value
    outDataset.GetRasterBand(1).WriteArray(arrayCopy) #Write the new array to it

    #Calculate the statistics
    outDataset.GetRasterBand(1).ComputeStatistics(0)

    #Populate the georeferencing info, etc
    outDataset.SetGeoTransform(templateRaster.GetGeoTransform())
    outDataset.SetProjection(templateRaster.GetProjection())
    outDataset.SetSpatialRef(templateRaster.GetSpatialRef())
    outDataset.SetMetadata(templateRaster.GetMetadata())

    drvr = None

    #Clean up file connections
    if not(doReturnGdalSource):
        outDataset = None

    #Close the template raster if it wasn't already loaded
    if doClose:
        templateRaster = None

    return outDataset 

def estimate_z_factor(rasterSrc:gdal.Dataset):
    """Estimate the multiplicative factor to make the dimensions of the elevations of the raster comparable
    to the dimensions of the raster's pixels

    Args:
        rasterSrc (gdal.Dataset): An open gdal raster dataset. For example, rasterSrc = gdal.Open(filename)

    Returns:
        zFactor (float): A number that represents the conversion necessary from the vertical units of the raster
            to the horizontal units defined by the projection. This will typically be 1 (e.g., both elevation measurements
            and coordinates are in meters) or something much less than 1 (e.g., elevation measurements are in meters, and
            coordinates degrees).
    """
    #Z factor estimates rely on the table found here:
    # https://www.esri.com/arcgis-blog/products/product/imagery/setting-the-z-factor-parameter-correctly/
    

    #First check if this is a geographic coordinate system, if not (and its projected) assume
    #a z_factor of 1
    zFactor = 1

    #Get the spatial reference of the raster
    spatial_ref = rasterSrc.GetSpatialRef()
    
    #If this is geographic
    if spatial_ref.IsGeographic():

        #And the angular units are degrees
        if spatial_ref.GetAngularUnitsName().lower() == 'degree':
            ##Define how to map coordinates to z_factors - necessary for approximating derivatives
            #When spatial coordinates are used that don't match elevation coordinates
            #This is an array: latitude, meters->degrees, feet -> degrees
            zFactorsTable = np.array([
            [0,0.00000898, 0.00000273],
            [10,0.00000912,	0.00000278],
            [20,0.00000956,	0.00000291],
            [30,0.00001036,	0.00000316],
            [40,0.00001171,	0.00000357],
            [50,0.00001395,	0.00000425],
            [60,0.00001792,	0.00000546],
            [70,0.00002619,	0.00000798],
            [80,0.00005156,	0.00001571],
            ])
            
            #Are we dealing with z unit in meters or feet.
            unit_index = 1 #1: Meters, 2: feet

            #Use the upper left coordinate of the raster to calculate z-value. A better option might be to
            # estimate the center of the raster, but that could be slightly more complicated if rasters were skewed
            uly  = rasterSrc.GetGeoTransform()[3] #Get the upper left y-coordinate of the raster

            #Interpolate as needed
            zFactor = np.interp(uly,zFactorsTable[:,0],zFactorsTable[:,unit_index])
        else:
            print('Warning: non-standard units detected in geographic coordinate system, proceeding with a z_factor of 1')

    return zFactor

def _apply_kernel_to_grid(grid:np.ndarray, kernel:np.ndarray, doUseConvolve:bool = True):
    """Convolves the supplied kernel with the grid. If doUseConvolve is True this applies the scipy.ndimage.convolve
    function, otherwise this uses scipy.ndimage.generic_filter - in some situations this may be faster.

    Args:
        grid (numpy.ndarray): Numpy array of the digital elevation model of interest
        kernel (numpy.ndarray): Numpy array of the kernel to apply
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.

    Returns:
        outputGrid (numpy.ndarray): The result of the convolution
    """
    if doUseConvolve:
        #Preform the convolution
        outputGrid = ndi.convolve(grid,kernel,**NDICONVOLVE_KWARGS)
    else:
        #Preform as an iteration
        sub_fun = lambda subarray: np.sum(kernel.flatten()*subarray.flatten())
        outputGrid = ndi.generic_filter(grid,sub_fun,kernel.shape,**NDICONVOLVE_KWARGS)

    return outputGrid 

def _get_windowed_mean_square_array(rasterGrid:np.ndarray, windowRadius:int = 1,
                          doUseConvolve:bool = True):
    """Calculates a windowed mean in a square area, where the window is of size
    (2*windowRadius + 1,2*windowRadius + 1) so that it is always centered about
    the pixel of interest.

    Args:
        rasterGrid (numpy.ndarray): Numpy array of the digital elevation model of interest
        windowRadius (int, optional): Defines the window shape as (2*windowRadius + 1,2*windowRadius + 1).
            Defaults to 1.
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.

    Returns:
        meanRaster (numpy.ndarray): The result of applying the square windowed mean filter to the input grid
    """

    #build kernel
    meanKernel = np.ones((2*windowRadius + 1,2*windowRadius + 1))/((2*windowRadius + 1)**2)

    #Preform the convolution
    meanRaster = _apply_kernel_to_grid(rasterGrid,meanKernel,doUseConvolve=doUseConvolve)

    return meanRaster


def _get_windowed_mean_circular_array(rasterGrid:np.ndarray, windowRadius:int = 1,
                            doUseConvolve:bool = True):
    
    """Calculates a windowed mean in a circular area, where the window is of size
    (2*windowRadius + 1,2*windowRadius + 1) so that it is always centered about
    the pixel of interest.

    Args:
        rasterGrid (numpy.ndarray): Numpy array of the digital elevation model of interest
        windowRadius (int, optional): Defines the window shape as (2*windowRadius + 1,2*windowRadius + 1).
            Defaults to 1.
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.

    Returns:
        meanRaster (numpy.ndarray): The result of applying the circular windowed mean filter to the input grid
    """
    #build a circular mean kernel
    X,Y = np.meshgrid(np.arange(-windowRadius,windowRadius+1), np.arange(-windowRadius,windowRadius+1)) #coordinates from center of kernel
    dist = np.sqrt(X**2 + Y**2) #distance from center of kernel
    meanKernel = 1.0*(dist <= windowRadius+0.5) #ones where cell center is at or less than radius
    meanKernel/=np.sum(meanKernel) #divide kernel by number of non-zero cells

    #Preform the convolution
    meanRaster = _apply_kernel_to_grid(rasterGrid,meanKernel,doUseConvolve=doUseConvolve)

    return meanRaster

def windowed_mean(savePath: str, inputRaster, windowRadius:int = 1,
                  doUseCircularWindow:bool = False, doUseConvolve:bool = True,
                  doReturnGdalSource:bool = False):
    """Calculates a windowed mean on the dataset of interest. The window is of size
    (2*windowRadius + 1,2*windowRadius + 1) so that it is always centered about
    the pixel of interest, but may be a circle or a square (default) depending
    on the value of doUseCircularWindow.

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        windowRadius (int, optional): Defines the window shape as (2*windowRadius + 1,2*windowRadius + 1).
            Defaults to 1.
        doUseCircularWindow (bool, optional): Should a circular (True) or a square (False) window
            be used? Defaults to False.
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the filter.

    """
    #Get the raster grid
    rasterGrid = get_raster_as_grid(inputRaster)[0]

    if doUseCircularWindow:
        meanRasterGrid = _get_windowed_mean_circular_array(rasterGrid,windowRadius=windowRadius, doUseConvolve = doUseConvolve)
    else:
        meanRasterGrid = _get_windowed_mean_square_array(rasterGrid,windowRadius=windowRadius, doUseConvolve = doUseConvolve)

    #Save the output dataset
    outDataset = duplicate_raster_with_array(meanRasterGrid,inputRaster,savePath,doReturnGdalSource)

    return outDataset

def less_windowed_mean(savePath:str, inputRaster, windowRadius:int = 1,
                       doUseCircularWindow:bool = False, doUseConvolve:bool = True,
                       doReturnGdalSource:bool = False):
    """Calculates the difference between the input raster and the mean in the requested window (raster - mean).
    The window is of size (2*windowRadius + 1,2*windowRadius + 1) so that it is always
    centered about the pixel of interest, but may be a circle or a square (default)
    depending on the value of doUseCircularWindow.

    This is a useful derivative for identifying edges and other 'high-frequency' content in a raster.

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        windowRadius (int, optional): Defines the window shape as (2*windowRadius + 1,2*windowRadius + 1).
            Defaults to 1.
        doUseCircularWindow (bool, optional): Should a circular (True) or a square (False) window
            be used? Defaults to False.
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the filter.

    """
    #Get the raster grid
    rasterGrid = get_raster_as_grid(inputRaster)[0]

    if doUseCircularWindow:
        meanRasterGrid = _get_windowed_mean_circular_array(rasterGrid,windowRadius=windowRadius, doUseConvolve = doUseConvolve)
    else:
        meanRasterGrid = _get_windowed_mean_square_array(rasterGrid,windowRadius=windowRadius, doUseConvolve = doUseConvolve)

        
    #subtract the two
    less_mean_array = rasterGrid - meanRasterGrid

    #Save the output dataset
    outDataset = duplicate_raster_with_array(less_mean_array,inputRaster,savePath,doReturnGdalSource)

    return outDataset


def _get_gaussian_mean_array(rasterGrid:np.ndarray, sigma:float, truncateStds:float,
                             doUseConvolve:bool):
    """Calculates a windowed mean weighted by a gaussian distribution with a standard deviation
    (in pixel units) of sigma and a window size that extends radially out to truncateStds*sigma.

    Args:
        rasterGrid (numpy.ndarray): Numpy array of the digital elevation model of interest
        sigma (float): standard deviation of gaussian distribution used to calculates weights for the
            weighted mean kernel.
        truncateStds (float): The number of standard deviations away from the central pixel that the weighted
            mean will be applied to. Creates a kernel that is of shape
            (2*int(sigma*truncateStds)+1, 2*int(sigma*truncateStds)+1)
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.

    Returns:
        gaussMeanRaster (numpy.ndarray): Grid representing the input rasterGrid after being smoothed by the 
            requested gaussian mean kernel.
        
    """
    #build a circular gaussian kernel
    radius = int(sigma*truncateStds)

    X,Y = np.meshgrid(np.arange(-radius,radius+1), np.arange(-radius,radius+1)) #coordinates from center of kernel
    dist = np.sqrt(X**2 + Y**2) #distance from center of kernel

    #Construct the gaussian kernel
    gauss_kernel = np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*((dist/sigma)**2))
    gauss_kernel/=np.sum(gauss_kernel)

    #Preform the convolution
    gaussMeanRaster = _apply_kernel_to_grid(rasterGrid,gauss_kernel,doUseConvolve=doUseConvolve)

    return gaussMeanRaster

def gaussian_mean(savePath:str, inputRaster, sigma:float = 3, truncateStds:float = 4,
                   doUseConvolve:bool = True, doReturnGdalSource:bool = False):
    
    """ Calculates a windowed mean weighted by a gaussian distribution with a standard deviation
        (in pixel units) of sigma and a window size that extends radially out to truncateStds*sigma.

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        sigma (float): standard deviation of gaussian distribution used to calculates weights for the
            weighted mean kernel.
        truncateStds (float): The number of standard deviations away from the central pixel that the weighted
            mean will be applied to. Creates a kernel that is of shape
            (2*int(sigma*truncateStds)+1, 2*int(sigma*truncateStds)+1)
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the filter.
    """
    #Load the raster as a grid
    rasterGrid = get_raster_as_grid(inputRaster)[0]

    #Apply the gaussian mean filter
    gaussMeanRaster = _get_gaussian_mean_array(rasterGrid,sigma,truncateStds,doUseConvolve)

    #Save the output dataset
    outDataset = duplicate_raster_with_array(gaussMeanRaster,inputRaster,savePath,doReturnGdalSource)

    return outDataset

def less_gaussian_mean(savePath:str, inputRaster, sigma:float = 3,
                       truncateStds = 4, doUseConvolve = True, doReturnGdalSource = False):
    
    """Calculates the difference between the input raster and that raster (raster - mean)
    after applying a windowed mean weighted by a gaussian distribution with a standard deviation
    (in pixel units) of sigma and a window size that extends radially out to truncateStds*sigma.

    This is a useful derivative for identifying edges and other 'high-frequency' content in a raster.

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        sigma (float): standard deviation of gaussian distribution used to calculates weights for the
            weighted mean kernel.
        truncateStds (float): The number of standard deviations away from the central pixel that the weighted
            mean will be applied to. Creates a kernel that is of shape
            (2*int(sigma*truncateStds)+1, 2*int(sigma*truncateStds)+1)
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the filter.
    """
    #Load the raster as a grid
    rasterGrid = get_raster_as_grid(inputRaster)[0]

    #Apply the gaussian mean filter
    gaussMeanRaster = _get_gaussian_mean_array(rasterGrid,sigma,truncateStds,doUseConvolve)

    #subtract the two
    lessGaussMeanArray = rasterGrid - gaussMeanRaster

    #Save the output dataset
    outDataset = duplicate_raster_with_array(lessGaussMeanArray,inputRaster,savePath,doReturnGdalSource)

    return outDataset


def difference_of_gaussian_means(savePath:str, inputRaster, sigma:float = 2, second_gaussian_multiplier:float = 3,
                       truncateStds = 4, doUseConvolve = True, doReturnGdalSource = False):
    """Calculates two gaussian mean rasters and differences them. The first is calculated based on a standard deviation
    of sigma, the second is calculated based on a standard deviation of sigma*second_gaussian_multiplier (a positive value
    greater than 1). The result is the difference of the first, calculated with a narrow standard deviation, from the second,
    calculated with a broad standard deviation.
    
    This is a band-pass filtering approach useful for edge detection.

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        sigma (float): standard deviation of gaussian distribution used to calculates weights for the
            weighted mean kernel.
        second_gaussian_multiplier (float, optional): The multiplier on the standard deviation on the second
            gaussian that is subtracted from the first. Defaults to 3. Should be greater than 1.
        truncateStds (float): The number of standard deviations away from the central pixel that the weighted
            mean will be applied to. Creates a kernel that is of shape
            (2*int(sigma*truncateStds)+1, 2*int(sigma*truncateStds)+1)
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the filter.
    """
    #Load the raster as a grid
    rasterGrid = get_raster_as_grid(inputRaster)[0]

    #Calculate the first gaussian mean filter
    gaussMeanRaster_1 = _get_gaussian_mean_array(rasterGrid,sigma,truncateStds,doUseConvolve)

    #Calculate the second gaussian mean filter
    gaussMeanRaster_2 = _get_gaussian_mean_array(rasterGrid,sigma*second_gaussian_multiplier,truncateStds,doUseConvolve)

    #subtract the two
    difference_of_gaussians = gaussMeanRaster_1 - gaussMeanRaster_2

    #Save the output dataset
    outDataset = duplicate_raster_with_array(difference_of_gaussians,inputRaster,savePath,doReturnGdalSource)

    return outDataset

def _get_slope_magnitude_array(rasterGrid:np.ndarray, nPixelRadius:int = 1,
                               dx:float = 1, dy:float=-1, zFactor:float = 1,
                               doUseConvolve:bool = True): 
    """Calculate a finite difference approximation of the magnitude of the slope on a
    2d numpy array. Uses a standard second-ordered centered approximation of slope if nPixelRadius is 1.
    In general, measures the slope by differencing the values of nPixelRadius in front of and behind each
    cell (e.g., dz_i/dx = (Z_i+nPixelRadius - Z_i-nPixelRadius)/(2*dx*nPixelRadius))

    Args:
        rasterGrid (numpy.ndarray): The digital elevation model represented as a numpy array.
        nPixelRadius (int, optional): The distance away from the central pixel to use
            in the finite difference approximation. Larger values will average slope over
            a greater distance. Defaults to 1.
        dx (float, optional): The cell size in the x (e.g., easting) direction. Defaults to 1.
        dy (float, optional): The cell size in the y (e.g., northing) direction. Defaults to 1.
        zFactor (float, optional): A number that represents the conversion necessary from the vertical units of the raster
            to the horizontal units defined by the projection. Defaults to 1.
        doUseConvolve (bool, optional): doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.

    Returns:
        slopeMagArray (numpy.ndarray): The magnitude of slope estimated with the requested centered difference kernel.
    """
    #Transform dx,dy based on the zFactor
    dx/=zFactor
    dy/=zFactor

    #Transform dx, dy based on the nPixelRadius
    dx*=2*nPixelRadius
    dy*=2*nPixelRadius

    #Initialize the kernels in the x and y directions
    Sx_kernel = np.zeros((nPixelRadius*2 + 1, nPixelRadius*2 + 1))
    Sy_kernel = np.zeros_like(Sx_kernel)
    
    #create kernel and perform the convolution
    Sx_kernel[nPixelRadius,0] = -1/dx
    Sx_kernel[nPixelRadius,-1] = 1/dx

    #create kernel and perform the convolution
    Sy_kernel[0,nPixelRadius] = -1/dy
    Sy_kernel[-1,nPixelRadius] = 1/dy

    Sx=_apply_kernel_to_grid(rasterGrid,Sx_kernel,doUseConvolve)
    Sy=_apply_kernel_to_grid(rasterGrid,Sy_kernel,doUseConvolve)

    #Calculate the magnitude
    slopeMagArray = np.sqrt(Sx**2 + Sy**2)

    return slopeMagArray

def slope(savePath:str, inputRaster, nPixelRadius:int = 1, doUseConvolve:bool = True,
           doReturnGdalSource:bool = False):
    
    """Calculates a finite difference approximation of the magnitude of the slope. Uses
    a second-ordered centered approximation of slope if nPixelRadius is 1. In general,
    measures the slope by differencing the values of nPixelRadius in front of and behind each
    cell (e.g., dz_i/dx = (Z_i+nPixelRadius - Z_i-nPixelRadius)/(2*dx*nPixelRadius))


    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        nPixelRadius (int, optional): The distance away from the central pixel to use
            in the finite difference approximation. Larger values will average slope over
            a greater distance. Defaults to 1.
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the kernel.
    """
    #Force nPixelRadius to be an integer
    nPixelRadius = int(nPixelRadius)

    #Get the raster grid
    rasterGrid,dx,dy,zFactor = get_raster_as_grid(inputRaster)

    #Get the numpy ndarray of the raster grid
    slopeMag = _get_slope_magnitude_array(rasterGrid,nPixelRadius,
                               dx, dy, zFactor,doUseConvolve)

    #Convert this to a geospatial dataset
    outDataset = duplicate_raster_with_array(slopeMag,inputRaster,savePath,doReturnGdalSource)

    return outDataset

def _get_aspect_magnitude_array(rasterGrid:np.ndarray, nPixelRadius:int = 1,
                               dx:float = 1, dy:float=-1, zFactor:float = 1,
                               doUseConvolve:bool = True, inDegrees=True):
    """Calculate a finite difference approximation of the magnitude of the aspect on a
    2d numpy array. Uses a standard second-ordered centered approximation of aspect if nPixelRadius is 1.
    In general, measures the aspect by differencing the values of nPixelRadius in front of and behind each
    cell (e.g., dz_i/dx = (Z_i+nPixelRadius - Z_i-nPixelRadius)/(2*dx*nPixelRadius))

    Args:
        rasterGrid (numpy.ndarray): The digital elevation model represented as a numpy array.
        nPixelRadius (int, optional): The distance away from the central pixel to use
            in the finite difference approximation. Larger values will average slope over
            a greater distance. Defaults to 1.
        dx (float, optional): The cell size in the x (e.g., easting) direction. Defaults to 1.
        dy (float, optional): The cell size in the y (e.g., northing) direction. Defaults to -1.
        zFactor (float, optional): A number that represents the conversion necessary from the vertical units of the raster
            to the horizontal units defined by the projection. Defaults to 1.
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        inDegrees (bool,optional): Returns the result in degrees (0-360). When False, returns the result
            in radians (-pi to pi).

    Returns:
        aspectMagArray (numpy.ndarray): The magnitude of aspect estimated with the requested centered difference kernel.
    """
    #Transform dx,dy based on the zFactor
    dx/=zFactor
    dy/=zFactor

    #Transform dx, dy based on the nPixelRadius
    dx*=2*nPixelRadius
    dy*=2*nPixelRadius

    #Initialize the kernels in the x and y directions
    Ax_kernel = np.zeros((nPixelRadius*2 + 1, nPixelRadius*2 + 1))
    Ay_kernel = np.zeros_like(Ax_kernel)
    
    #create kernel and perform the convolution
    Ax_kernel[nPixelRadius,0] = -1/dx
    Ax_kernel[nPixelRadius,-1] = 1/dx

    #create kernel and perform the convolution
    Ay_kernel[0,nPixelRadius] = -1/dy
    Ay_kernel[-1,nPixelRadius] = 1/dy

    Ax=_apply_kernel_to_grid(rasterGrid,Ax_kernel,doUseConvolve)
    Ay=_apply_kernel_to_grid(rasterGrid,Ay_kernel,doUseConvolve)

    #Calculate the magnitude
    if inDegrees:
        aspectMagArray=((np.arctan2(Ay,Ax)*180/np.pi)-90) % -360
        aspectMagArray*=-1
    else:
        aspectMagArray=np.arctan2(Ay,Ax)

    return aspectMagArray

def aspect(savePath:str, inputRaster, nPixelRadius:int = 1, doUseConvolve:bool = True,
           doReturnGdalSource:bool = False, inDegrees=True):
    """Calculates a finite difference approximation of the aspect. Uses
    a second-ordered centered approximation of aspect if nPixelRadius is 1. In general,
    measures the aspect by differencing the values of nPixelRadius in front of and behind each
    cell (e.g., dz_i/dx = (Z_i+nPixelRadius - Z_i-nPixelRadius)/(2*dx*nPixelRadius))

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        nPixelRadius (int, optional): The distance away from the central pixel to use
            in the finite difference approximation. Larger values will average slope over
            a greater distance. Defaults to 1.
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.
        inDegrees (bool,optional): Returns the result in degrees (0-360). When False, returns the result
            in radians (-pi to pi).

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the kernel.
    """
    #Force nPixelRadius to be an integer
    nPixelRadius = int(nPixelRadius)

    #Get the raster grid
    rasterGrid,dx,dy,zFactor = get_raster_as_grid(inputRaster)

    #Get the numpy ndarray of the raster grid
    aspectMag = _get_aspect_magnitude_array(rasterGrid,nPixelRadius,
                               dx, dy, zFactor,doUseConvolve, inDegrees)


    #Convert this to a geospatial dataset
    outDataset = duplicate_raster_with_array(aspectMag,inputRaster,savePath,doReturnGdalSource)

    return outDataset

def _get_laplacian_array(rasterGrid:np.ndarray, nPixelRadius:int = 1,
                               dx:float = 1, dy:float=-1, zFactor:float = 1,
                               doUseConvolve:bool = True):
    """Calculates the surface laplacian with a centered difference kernel, measuring differences between
    the central pixel and pixels nPixelRadius away in the x and y directions.
    
    For example, in one dimension d^2z_i/dx^2 = (2*z_i - z_(i-nPixelRadius) - z(i+nPixelRadius))/(dx*nPixelRadius)**2

    Args:
        rasterGrid (numpy.ndarray): The digital elevation model represented as a numpy array.
        nPixelRadius (int, optional): The distance away from the central pixel to use
            in the finite difference approximation. Larger values will average derivatives over
            a greater distance. Defaults to 1.
        dx (float, optional): The cell size in the x (e.g., easting) direction. Defaults to 1.
        dy (float, optional): The cell size in the y (e.g., northing) direction. Defaults to -1.
        zFactor (float, optional): A number that represents the conversion necessary from the vertical units of the raster
            to the horizontal units defined by the projection. Defaults to 1.
        doUseConvolve (bool, optional): doUseConvolve (bool, optional): Should ndimage.convolve be used?
            If False, instead iterates using scipy.ndimage.generic_filter. Defaults to True.

    Returns:
        laplacianArray (numpy.ndarray): The laplacian estimated with the requested centered difference kernel.
    """
    #Transform dx,dy based on the zFactor
    dx/=zFactor
    dy/=zFactor

    #Tranform dx,dy based on how wide the kernel is
    dx*=nPixelRadius
    dy*=nPixelRadius

    #Initialize the kernel
    curv_kernel = np.zeros((nPixelRadius*2 + 1, nPixelRadius*2 + 1))
    
    #create kernel and perform the convolution
    curv_kernel[nPixelRadius,nPixelRadius] = (2.0/dx**2) + (2.0/dy**2) #Center component has weights from differencs in x and y directions

    #From gradient in row (y) orientation
    curv_kernel[0,nPixelRadius] = -1.0/dy**2
    curv_kernel[-1,nPixelRadius] = -1.0/dy**2

    #From gradient in column (x) orientation
    curv_kernel[nPixelRadius,-1] = -1.0/dx**2
    curv_kernel[nPixelRadius,0] = -1.0/dx**2

    laplacianArray=_apply_kernel_to_grid(rasterGrid,curv_kernel,doUseConvolve)

    return laplacianArray

def laplacian(savePath:str, inputRaster, nPixelRadius:int = 1, doUseConvolve:bool = True,
               doReturnGdalSource:bool = False):
    """Calculates the surface laplacian with a centered difference kernel, measuring differences between
    the central pixel and pixels nPixelRadius away in the x and y directions.
    
    For example, in one dimension d^2z_i/dx^2 = (2*z_i - z_(i-nPixelRadius) - z(i+nPixelRadius))/(dx*nPixelRadius)**2

    This illuminates changes in slope, and as such is helpful for visualizing hilltops, valleys, escarpments,
    undulations in hillslopes associated with bedding, and so on.

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        nPixelRadius (int, optional): The distance away from the central pixel to use
            in the finite difference approximation. Larger values will average derivatives over
            a greater distance. Defaults to 1.
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
        will instead return a gdal.Dataset instance representing the input raster after application of the kernel.
    """
    #Force nPixelRadius to be an integer
    nPixelRadius = int(nPixelRadius)

    #Get the raster grid
    rasterGrid,dx,dy,zFactor = get_raster_as_grid(inputRaster)

    laplacianArray = _get_laplacian_array(rasterGrid,nPixelRadius,
                               dx, dy, zFactor,
                               doUseConvolve)

    #Convert this to a geospatial dataset
    outDataset = duplicate_raster_with_array(laplacianArray,inputRaster,savePath,doReturnGdalSource)

    return outDataset

def _get_tpi_array(rasterGrid:np.ndarray, outerPixelRadius:int = 10, innerPixelRadius:int = 3,
                   distanceWeightingExponent:float = 0, doUseConvolve:bool = True):
    """Calculate the topographic position index, the difference between the digital elevation model
    and the mean calculated in an annulus centered around each pixel.

    Args:
        rasterGrid (numpy.ndarray): The digital elevation model represented as a numpy array.
        outerPixelRadius (int, optional): The outer radius of the annulus, measured as a number of pixels.
            Defaults to 10.
        innerPixelRadius (int, optional): The inner radius of the annulus, measured as a number of pixels.
            Defaults to 3.
        distanceWeightingExponent (float, optional): The exponent to use in weighting values as a function
            of distance to calculate the annulus mean. Values less than 0 can be used to calculate a weighted mean
            that provides greater weight to pixels near the center of the annulus.
            Defaults to 0 (equivalent to no weighting by distance). 
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.

    Returns:
        tpiArray (np.ndarray): TPI calculated with the requested kernel.
    """
    #X,Y coordinates of the kernel from center
    X,Y = np.meshgrid(np.arange(-outerPixelRadius,outerPixelRadius+1),
                       np.arange(-outerPixelRadius,outerPixelRadius+1))
    
    #radial distance of kernel
    dist = np.sqrt(X**2 + Y**2)

    #Mask the distance
    in_bounds = (dist < outerPixelRadius) & (dist >=innerPixelRadius)
    
    #calculate the kernel as the inverse of distance raised to an exponent
    #if that exponent is 0 (the default), weights will be 1 for all cells in donut
    kernel = in_bounds*(dist**-distanceWeightingExponent)

    #normalize the kernel values
    kernel/=np.sum(kernel)

    #Perform the convolution
    donut_mean = _apply_kernel_to_grid(rasterGrid,kernel,doUseConvolve=doUseConvolve)

    #calculate difference the kernel mean and the local values
    tpiArray = rasterGrid - donut_mean

    return tpiArray


def tpi(savePath:str, inputRaster, outerPixelRadius:int = 10, innerPixelRadius:int = 3,
        distanceWeightingExponent:float = 0, doUseConvolve:bool = True,
        doReturnGdalSource:bool = False):
    
    """Calculate the topographic position index (TPI), the difference between the digital elevation model
    and the mean calculated in an annulus centered around each pixel.

    TPI is useful for revealing high-frequency content in a raster, and is very sensitive to input parameters.

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        outerPixelRadius (int, optional): The outer radius of the annulus, measured as a number of pixels.
            Defaults to 10.
        innerPixelRadius (int, optional): The inner radius of the annulus, measured as a number of pixels.
            Defaults to 3.
        distanceWeightingExponent (float, optional): The exponent to use in weighting values as a function
            of distance to calculate the annulus mean. Values less than 0 can be used to calculate a weighted mean
            that provides greater weight to pixels near the center of the annulus.
            Defaults to 0 (equivalent to no weighting by distance). 
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
    
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the kernel.
    """
    #Get the raster grid
    rasterGrid = get_raster_as_grid(inputRaster)[0]

    #Apply the tpi calculation
    tpiArray = _get_tpi_array(rasterGrid,outerPixelRadius, innerPixelRadius,distanceWeightingExponent, doUseConvolve=doUseConvolve)

    #Convert this to a geospatial dataset
    outDataset = duplicate_raster_with_array(tpiArray,inputRaster,savePath,doReturnGdalSource)

    return outDataset

def ricker_wavelet(savePath:str, inputRaster, sigma:float = 1.0, kernelRadius:int = 8, doUseConvolve:bool = True,
        doReturnGdalSource:bool = False):
    """Computes the 2D continuous wavelet transformation of a DEM using the Ricker wavelet. 
    This wavelet functions as a band-pass filter. It enhances specific topographic features 
    by magnifying variations at a designated spatial frequency, controlled by the parameter sigma.

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory' 
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        sigma (float, optional): Specifies the size of the wavelet that will be convolved across the target 
            DEM in units of meters. The wavelet can be any value >=1 and depends on the scale of the features that 
            you want to visualize. The smallest features (~1m) would be visualized with a sigma value of 1 and 
            increasing feature directly correlates to increased wavelet  Defaults to 1.0.
        kernelRadius (int, optional): Value to multiply sigma by to define the radius of the kernel.
            Defaults to 8.
        doUseConvolve (bool, optional): Should ndimage.convolve be used? If False, instead iterates
            using scipy.ndimage.generic_filter. Defaults to True.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False. 

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the kernel.
    """
    #Get the grid, dx, dy, and zFactor from the inputRaster
    rasterGrid,dx,dy,zFactor = get_raster_as_grid(inputRaster)

    #Transform dx,dy based on the zFactor si unit of dx and dy are effectively in meters.
    dx/=zFactor
    dy/=zFactor

    #Shorten kernel parameters
    k = kernelRadius
    a = sigma

    #Create mesh grids X and Y that define the spatial coordinates over which the kernel will be evaluated
    xIntRadius = np.abs(np.ceil(k*a/dx)) #Round to nearest integer
    yIntRadius = np.abs(np.ceil(k*a/dy)) #Round to nearest integer
    X,Y = np.meshgrid(np.arange(-xIntRadius, xIntRadius +1)*dx, np.arange(-yIntRadius, yIntRadius +1)*dy)

    #Calculate the kernel using the Ricker wavelet formula
    kernel = (1/(np.pi*a**4)) * (1 - (X**2 + Y**2)/(2*a**2)) * np.exp(-(X**2 + Y**2)/(2*a**2)) #units of [1/(m^4)]
    
    #Convolve dem with kernel and multiply the result by dx and dy to account for the pixel size in the convolution operation.
    C = _apply_kernel_to_grid(rasterGrid,kernel,doUseConvolve=doUseConvolve)*np.abs(dx*dy)

    #Convert this to a geospatial dataset
    outDataset = duplicate_raster_with_array(C,inputRaster,savePath,doReturnGdalSource)
  
    return outDataset

def absolute_value(savePath:str, inputRaster, doReturnGdalSource:bool = False):
    """Calculates the absolute value of an input raster

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the filter.

    """
    #Get the raster grid
    rasterGrid = get_raster_as_grid(inputRaster)[0]

    #take the absolute value and save
    outDataset=duplicate_raster_with_array(np.abs(rasterGrid),inputRaster,savePath,doReturnGdalSource)
    
    return outDataset    

def standard_deviation(savePath:str, inputRaster, windowRadius:int = 1, doReturnGdalSource:bool = False):
    """Calculates the standard deviation of an input raster within a moving window

    Args:
        savePath (str): The path to save the resulting dataset. Alternatively, may be 'memory'
            to just create a gdal.Dataset in memory.
        inputRaster (str OR gdal.Dataset): The path to the raster of interest or a gdal.Dataset that
            has already been loaded.
        windowRadius (int, optional): Defines the window shape as (2*windowRadius + 1,2*windowRadius + 1).
            Defaults to 1.
        doReturnGdalSource (bool, optional): Should a gdal.Dataset be returned (True), if not
            returns None. Defaults to False.

    Returns:
        outDataset (None OR gdal.Dataset): If doReturnGdalSource is False, returns None. If doReturnGdalSource is True
            will instead return a gdal.Dataset instance representing the input raster after application of the filter.

    """
    #Get the raster grid
    rasterGrid = get_raster_as_grid(inputRaster)[0]

    #make the kernel
    kernel=np.ones((2*windowRadius + 1,2*windowRadius + 1))/((2*windowRadius + 1)**2)

    #take the standard deviation
    outputGrid = ndi.generic_filter(rasterGrid,np.std,footprint=kernel,**NDICONVOLVE_KWARGS)

    #save the output dataset
    outDataset=duplicate_raster_with_array(outputGrid,inputRaster,savePath,doReturnGdalSource)

    return outDataset 

if __name__=='__main':
    pass

# %%
