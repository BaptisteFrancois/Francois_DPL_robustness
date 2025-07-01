import arcpy
import sys
sys.path.append('..')
import os
import arcDemTools as adt 

if __name__ == '__main__':
       
    # Tool parameter accessed with GetParameter or GetParameterAsText
    rasterPaths = arcpy.GetParameterAsText(0)
    #arcpy includes unecessary quotes; delete them
    if "'" in rasterPaths:
        rasterPaths=rasterPaths.replace("'","")

    derivatives = arcpy.GetParameterAsText(1)

    #Turn derivatives into a list, accounting for colon delimited in arcpy.GetParamterAsText output
    if ';' in derivatives:
        derivatives = [x.strip() for x in derivatives.split(";")]
    else:
        derivatives = [derivatives]

    #Turn rasters into a list, accounting for colon delimited in arcpy.GetParamterAsText output
    if ';' in rasterPaths:
        rasterPaths = [x.strip() for x in rasterPaths.split(";")]
    else:
        rasterPaths = [rasterPaths]

    #Save these in the same directory that rasterPath is stored in
    outputPath = [os.path.split(x)[0] for x in rasterPaths]

    #Calculate specific derivatives
    filesToAddToArc = adt.arc_build_derivatives(rasterPaths, derivatives, outputPath)
            
    #If anything was downloaded, add it to the map
    if filesToAddToArc:
        arcpy.SetProgressorLabel("Adding rasters to map")  
        adt.format_add_raster_data(filesToAddToArc, doStats=True, doPyramids=True)