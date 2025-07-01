'''
These defaults can be edited by the user before being passed to 
the compute_derivatives function. For a full list of options, see 
https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.DEMProcessingOptions
'''
from osgeo import gdal
from . import dem_derivatives as dd

#This dictionary provides a dictionary of input arguments that control the computation
#of derivatives
derivative_params = {

    'absolute_value': {
        #No arguments for absolute value, but it needs a default dictionary to function
    },

    'aspect' : {
        'nPixelRadius':1, #number of pixels away from central pixel to use when calculating centered difference
    },

    'difference_of_gaussian_means':{
        'sigma':2,#standard deviation to use for gaussian weighting (in pixels)
        'second_gaussian_multiplier':3, #Multiplier on sigma used for the second gaussian mean that is subtracted from the first
        'truncateStds' : 4, #Number of standard deviations away from center to consider in kernel
    },

    'gaussian_mean' : {
        'sigma' : 3, #standard deviation to use for gaussian weighting (in pixels)
        'truncateStds' : 4, #Number of standard deviations away from center to consider in kernel
    },

    'hillshade': {
        'processing':'hillshade',
        'azimuth': 315, #Change to a value from 0-360 to update light direction
        'altitude':45, #Change to a value from 0-90 to update light angle
        'multiDirectional' : False, #Change to True for a multi-directional hillshade
        'computeEdges':False,
        #Adjust this to the appropriate value if you are using a geographic coordinate system
        #NOTE! If you are using the arcpro tools or the dem_getter.compute_derivatives function
        #zFactor will be estimated for you if you leave this as 'none'
        'zFactor': None, 
    },

     'hillshade_nadir': {
        'processing':'hillshade',
        'azimuth': 315, #Change to a value from 0-360 to update light direction
        'altitude':90, #nadir hillshade ignores light orientation, is just a slope transform
        'multiDirectional' : False, #Change to True for a multi-directional hillshade
        'computeEdges':False,
        #Adjust this to the appropriate value if you are using a geographic coordinate system
        #NOTE! If you are using the arcpro tools or the dem_getter.compute_derivatives function
        #zFactor will be estimated for you if you leave this as 'none'
        'zFactor': None, 
    },

    'laplacian': {
        #What distance away from the central pixel do we want to calculate the laplacian based on?
        #Bigger numbers will result in a smoother result by averaging out noise, but will not reveal
        #edges as sharply. 1 is the true definition of a second-order finite difference approximation.
        'nPixelRadius' : 1,
    },

    'less_gaussian_mean': {
        'sigma' : 2, #standard deviation to use for gaussian weighting (in pixels)
        'truncateStds' : 4, #Number of standard deviations away from center to consider in kernel
    },

    'less_windowed_mean':{
        'windowRadius' : 1, #Width of moving window in pixels
        'doUseCircularWindow' : False,#Change to 'True' to use a circular instead of a square window
    },

    'ricker_wavelet':{
        'sigma':1.0, #wavelet scale, determines the size of feature captured in transformation
        'kernelRadius':8 #radius of the kernel used during convolutions
    },

    'roughness':{
        'processing':'Roughness',
        'computeEdges':True
    },

    'slope' : {
        'nPixelRadius':1, #number of pixels away from central pixel to look
    },

    'TPI' :{
        'outerPixelRadius': 10, #outer radius of annulus within which mean is calculated
        'innerPixelRadius': 3, #inner radius of annulus within whcih mean is callculated
        
        #some calculations of TPI allow for the mean value in the annulus to be calculated by weighting
        #cell values based on inverse distance. This exponent scales that weighting. 0: all pixels in the annulus
        # have the same weight, 1: weights are assigned according to the inverse distance, 2: weights are assigned
        #according to the inverse of distance squared
        'distanceWeightingExponent' : 0
    },

    'windowed_mean': {
        'windowRadius' : 1, #Width of moving window in pixels
        'doUseCircularWindow' : False,#Change to 'True' to use a circular instead of a square window
    },

    'standard_deviation': {
        'windowRadius' : 1 #Length/width of the moving window in pixels

    }

}

#This dictionary maps a plain text name, to a function that can be used (when paired with derivative defaults)
#To compute a topographic derivative
derivative_functions = {
    'gaussian_mean': dd.gaussian_mean,
    'hillshade': gdal.DEMProcessing,
    'hillshade_nadir': gdal.DEMProcessing,
    'difference_of_gaussian_means':dd.difference_of_gaussian_means,
    'laplacian':dd.laplacian,
    'less_gaussian_mean': dd.less_gaussian_mean,
    'less_windowed_mean' : dd.less_windowed_mean,
    'ricker_wavelet': dd.ricker_wavelet,
    'roughness':gdal.DEMProcessing,
    'slope': dd.slope,
    'TPI':dd.tpi,
    'windowed_mean': dd.windowed_mean,
    'aspect': dd.aspect,
    'standard_deviation':dd.standard_deviation,
    'absolute_value':dd.absolute_value,
}

