import arcpy
import sys
sys.path.append('..')
from dem_getter import dem_getter as dg
import arcDemTools as adt

def getBboxData(datasetName, xmin, ymin, xmax, ymax, filePath, dataType, inputEPSG, excludeRedundant):

    fileList=dg.get_aws_paths(dataset=datasetName, xMin=xmin, yMin=ymin, xMax=xmax, yMax=ymax, filePath = None,
    dataType = dataType, inputEPSG=inputEPSG, doExcludeRedundantData=excludeRedundant)
    
    if fileList:
        
        dlPaths = adt.arc_batch_download(inFiles=fileList, filePath=filePath)

        return dlPaths
    else:
        return arcpy.AddWarning('No products available.')

# This is used to execute code if the file was run but not imported
if __name__ == '__main__':

    # Tool parameter accessed with GetParameter or GetParameterAsText
    dataset = arcpy.GetParameterAsText(0)
    bbox = arcpy.GetParameterAsText(1) 
    arcEPSG=arcpy.GetParameterAsText(2)
    filepath = arcpy.GetParameterAsText(3)
    datatype = arcpy.GetParameterAsText(4)
    mergePath=arcpy.GetParameter(5)
    doExcludeRedundant=arcpy.GetParameter(6)
    cleanUp=arcpy.GetParameter(7)
    derivatives=arcpy.GetParameter(8)

    allDownloads = set()  #Keep a set of files we've already downloaded with previous quads, at the end this will be all dls

    #get EPSG code from the input CRS
    spatial_ref=arcpy.SpatialReference()
    spatial_ref.loadFromString(str(arcEPSG))
    inEPSG=spatial_ref.factoryCode

    #get bbox coords in correct format
    bboxSplit=bbox.split(' ')
    xmin=float(bboxSplit[0])
    ymin=float(bboxSplit[1])
    xmax=float(bboxSplit[2])
    ymax=float(bboxSplit[3])

    #download data, save download path
    dlPaths=getBboxData(dataset, xmin, ymin, xmax, ymax, filepath, datatype, inEPSG, doExcludeRedundant)

    #If any files were downloaded, proceed. Else, note that nothing was found.
    if dlPaths:
        allDownloads = allDownloads.union(dlPaths)

        if dataset =='NED_1-9as' or datatype=="IMG":
            import zipfile

            ext=('.img')
            zipPaths=[]
            for dl in dlPaths:
                with zipfile.ZipFile(dl,'r') as zippie:
                    zipPath=[zippie.extract(file, filepath) for file in zippie.namelist() if file.endswith(ext)]
                    zipPaths.append(zipPath[0])
                
            dlPaths=zipPaths
            allDownloads = allDownloads.union(zipPaths)

        #warn the user if they are attempting to download more than 500 files
        if len(dlPaths) == dg.MAXITEMS:
            arcpy.AddWarning('This request hit an internal limit on the number of files ({} files), potentially leaving gaps in your request.\n Please try a smaller request or update the value of MAXITEMS in the dem_getter.py file.'.format(str(dg.MAXITEMS)))

        #If a merge was requested...
        if mergePath:
            #Reset the progress bar
            arcpy.ResetProgressor()
            arcpy.SetProgressor("default", message="Merging rasters")

            #set merge extent to user input bounding box
            mergeExtent=([xmin, xmax], [ymin,ymax])

            #get outFileName in correct format
            outFile = dg.os.path.join(filepath,'{}_'.format(dataset)+mergePath) 

            #Merge and warp the dems
            dg.merge_warp_dems(dlPaths, outFile, outExtent = mergeExtent, outEPSG = inEPSG, pixelSize=None)

            #If the user requested cleanup of the intermediary files
            if cleanUp:
                #The files needed for this download, in the expected path, including those previous downloaded
                downloadedPaths = [dg.os.path.join(filepath,line.strip().split('/')[-1]) for line in allDownloads]
                for file in downloadedPaths:
                    arcpy.SetProgressorLabel("Cleaning up intermediary data")        
                    dg.os.remove(file)
                    #arcpy.management.Delete(file)  

                dlPaths = [] #Replace the all rasters to load file with an empty list (we'll add the merge if below)
                
            dlPaths.append(outFile) 
            
        if cleanUp:
            arcpy.AddMessage('Clean up intermediary was true')

        if derivatives:
            #Make derivatives for desired paths
            derivFiles = adt.arc_build_derivatives(dlPaths, derivatives, filepath)
            dlPaths.extend(derivFiles)

        #Add all desired outputs to map
        adt.format_add_raster_data(dlPaths, doStats = True, doPyramids=True)

    else:
        arcpy.AddMessage('No available datasets were found, please consult documentation and confirm that you entered in parameters correctly')