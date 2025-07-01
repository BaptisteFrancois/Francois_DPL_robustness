import arcpy
import sys
sys.path.append('..')
from dem_getter import dem_getter as dg
import arcDemTools as adt 

# This is used to execute code if the file was run but not imported
if __name__ == '__main__':
    
    # Tool parameter accessed with GetParameter or GetParameterAsText
    dataset = arcpy.GetParameterAsText(0)
    quad = arcpy.GetParameterAsText(1)
    #quad names with more than one word are read in with extra quotes;
    #get rid of them
    if "'" in quad:
        quad=quad.replace("'","")

    state=arcpy.GetParameterAsText(2)
    #might not work for Alaska, warn user
    if state=='Alaska':
        arcpy.AddWarning('Warning: some datasets in Alaska may be unavailable to fetch using this tool.')

    filepath = arcpy.GetParameterAsText(3)
    datatype = arcpy.GetParameterAsText(4)
    mergeExt=arcpy.GetParameter(5)
    #join the dataset name to the merge extension
    mergePath=dg.os.path.join(dataset+'{}'.format(mergeExt))
    
    doExcludeRedundant=arcpy.GetParameter(6)
    cleanUp=arcpy.GetParameter(7)
    derivatives=arcpy.GetParameter(8)

    #If quad is delimited or not turn it into a list
    if ';' in quad:
        quad = [x.strip() for x in quad.split(";")]
    else:
        quad = [quad]
        
    #Preallocate some lists to keep track of the merged datasets we want to preserve, the files downloaded
    filesToAddToArc = [] #A set of the files we want to add to arc (e.g., merged quads)
    #allDownloads is a set, sets can't have repeated values
    allDownloads = set()  #Keep a set of files we've already downloaded with previous quads, at the end this will be all dls
    
    #Loop through each of the quads
    for quad_i in quad:
            
        #Find the quads to download
        dl=dg.get_aws_paths_from_24kQuadName(dataset=dataset, quadName=quad_i, stateName=state, filePath=None,
                                        dataType=datatype, doExcludeRedundantData=doExcludeRedundant)
        
        #Add them to a list of all quads
        if dl:
            newDownloads = set(dl).difference(allDownloads) #What are the new downloads we need?
            
            #Download just the new downloads
            downloadPaths=adt.arc_batch_download(inFiles=newDownloads, filePath=filepath)
            allDownloads = allDownloads.union(downloadPaths) #union on a set will give back a set of both items

            if dataset =='NED_1-9as' or datatype=="IMG":
                import zipfile

                ext=('.img')
                zipPaths=[]
                for dl in downloadPaths:
                    with zipfile.ZipFile(dl,'r') as zippie:
                        zipPath=[zippie.extract(file, filepath) for file in zippie.namelist() if file.endswith(ext)]
                        zipPaths.append(zipPath[0])

                downloadPaths=zipPaths
                allDownloads = allDownloads.union(zipPaths)

            #Merge these into the quad boundary
            arcpy.ResetProgressor()
            arcpy.SetProgressor("default", message="Merging rasters for quad: {}".format(quad_i))

            outFile, inEPSG=adt.format_for_quad_merge(quad_i, filepath, mergePath, downloadPaths[0])

            #get quad extent for merging
            mergeExtent=adt.get_quad_extent(quad_i, state, inEPSG)
            
            dg.merge_warp_dems(downloadPaths, outFile, outExtent=mergeExtent)
            filesToAddToArc.append(outFile)

            if derivatives:
                derivs = adt.arc_build_derivatives([outFile], derivatives, filepath)
                filesToAddToArc.extend(derivs)
            
        else:
            arcpy.AddWarning('No {} products available for {}.'.format(dataset,quad_i))
        
    if cleanUp:
        #The files needed for this download, in the expected path, including those previous downloaded
        downloadedPaths = [dg.os.path.join(filepath,line.strip().split('/')[-1]) for line in allDownloads]
        for file in downloadedPaths:
            arcpy.SetProgressorLabel("Cleaning up intermediary data")        
            dg.os.remove(file)
    
    #If anything was downloaded, add it to the map
    if filesToAddToArc:
        arcpy.SetProgressorLabel("Adding rasters to map")  
        adt.format_add_raster_data(filesToAddToArc, doStats=True, doPyramids=True)

    
    # # Update derived parameter values using arcpy.SetParameter() or arcpy.SetParameterAsText()