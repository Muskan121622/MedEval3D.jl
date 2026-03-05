"""
this kernel will prepare da
"""
module PrepareArrtoBool
using KernelAbstractions, Atomix, Logging,..CUDAGpuUtils, StaticArrays, ..IterationUtils,..BitWiseUtils, ..ReductionUtils, ..CUDAAtomicUtils,..MetaDataUtils,..HFUtils
export getBoolCube_kernel,getLargeForBoolKernel,getSmallForBoolKernel,@planeIter,@localAllocations,@uploadLocalfpFNCounters,@uploadMinMaxesToShmem,@uploadDataToMetaData,@finalGlobalSet


"""
allocates in the local, register in shared memory
"""
macro localAllocations()

    return esc(quote
    anyPositive = false # true If any bit will bge positive in this array - we are not afraid of data race as we can set it multiple time to true
    
    # In KA, we'll use @localmem. We need to know the sizes. 
    # Assuming dataBdim is available in the scope where this macro is called.
    locFps = UInt32(0)
    locFns = UInt32(0)
    offsetIter = UInt8(0)
    
    # Dynamic shared memory replacement. 
    # For now, we'll use a reasonably large fixed size or expect it to be passed.
    # Since dataBdim is typically 32x32, we can use (32, 32).
    shmemblockData = @localmem(UInt32, (32, 32))

    # Resetting shmemblockData
    shmemblockData[@index(Local, X), @index(Local, Y)] = 0

    ######## needed for establishing min and max values of blocks that are intresting us 
    minX = @localmem(Float32, 1)
    maxX = @localmem(Float32, 1)
    minY = @localmem(Float32, 1)
    maxY = @localmem(Float32, 1)
    minZ = @localmem(Float32, 1)
    maxZ = @localmem(Float32, 1)      
     
    # Resetting on first thread of block
    if @index(Local, Linear) == 1
        minX[1] = Float32(1110.0)
        maxX[1] = Float32(0.0)
        minY[1] = Float32(1110.0)
        maxY[1] = Float32(0.0)    
        minZ[1] = Float32(1110.0)
        maxZ[1] = Float32(0.0) 
    end

    localQuesValues = @localmem(UInt32, 14)   
    
    # Initialize localQuesValues
    if @index(Local, Linear) <= 14
        localQuesValues[@index(Local, Linear)] = 0
    end
     
    @synchronize
end)
end



"""
invoked on each lane and on the basis of its position will update the number of fp or fn in given queue
"""
macro uploadLocalfpFNCounters()
   return esc(quote
   atomicallyAddToSpot(localQuesValues,getIndexOfQueue(xpos,ypos,zpos,dataBdim,boolSegm),1)
    end)
end   

"""
invoked after we gone through data block and now we save data into shared memory
"""
macro uploadMinMaxesToShmem()
    return  esc(quote
        if anyPositive
            Atomix.@atomic minX[1] = min(minX[1], Float32(xMeta + 1))
            Atomix.@atomic maxX[1] = max(maxX[1], Float32(xMeta + 1))
            Atomix.@atomic minY[1] = min(minY[1], Float32(yMeta + 1))
            Atomix.@atomic maxY[1] = max(maxY[1], Float32(yMeta + 1))
            Atomix.@atomic minZ[1] = min(minZ[1], Float32(zMeta + 1))
            Atomix.@atomic maxZ[1] = max(maxZ[1], Float32(zMeta + 1))
        end
    end)
end

"""
invoked after we gone through data block and now we save data into appropriate spots in metadata of this metadata block
"""
macro uploadDataToMetaData()
    esc(quote
        # Using Local, Linear for simplicity in clearing/uploading
        tid = @index(Local, Linear)
        if tid <= 14 && anyPositive
            @setMeta(getBeginingOfFpFNcounts() + tid, localQuesValues[tid])
        end
        
        @synchronize

        if tid == 15 && anyPositive
            total_fp = localQuesValues[1] + localQuesValues[3] + localQuesValues[5] + localQuesValues[7] + localQuesValues[9] + localQuesValues[11] + localQuesValues[13]
            @setMeta(getBeginingOfFpFNcounts() + 15, total_fp)
        end
        if tid == 16 && anyPositive
            total_fn = localQuesValues[2] + localQuesValues[4] + localQuesValues[6] + localQuesValues[8] + localQuesValues[10] + localQuesValues[12] + localQuesValues[14]
            @setMeta(getBeginingOfFpFNcounts() + 16, total_fn)
        end
    end)
end#uploadDataToMetaData

"""
invoked after all of the data was scanned so after we will do atomics between blocks we will know 
    the minimal and maximal in each dimensions
"""
macro finalGlobalSet()
    esc(quote
        # reduction part - assuming block size is compatible
        # this is a very manual reduction from the original code
        # we'll use Atomix for the final global writes
        tid = @index(Local, Linear)
        if tid == 1
            Atomix.@atomic fn[1] += locFns
            Atomix.@atomic fp[1] += locFps
            
            Atomix.@atomic minxRes[1] = min(minxRes[1], minX[1])
            Atomix.@atomic maxxRes[1] = max(maxxRes[1], maxX[1])

            Atomix.@atomic minyRes[1] = min(minyRes[1], minY[1])
            Atomix.@atomic maxyRes[1] = max(maxyRes[1], maxY[1])

            Atomix.@atomic minzRes[1] = min(minzRes[1], minZ[1])
            Atomix.@atomic maxzRes[1] = max(maxzRes[1], maxZ[1])
        end
    end)
end


"""
we need to give back number of false positive and false negatives and min,max x,y,x of block containing all data 
adapted from https://github.com/JuliaGPU/CUDA.jl/blob/afe81794038dddbda49639c8c26469496543d831/src/mapreduce.jl
goldGPU - array holding 3 dimensional data of gold standard bollean array
segmGPU - array with 3 dimensional the data we want to compare with gold standard
reducedGold - the smallest boolean block (3 dim array) that contains all positive entris from both masks
reducedSegm - the smallest boolean block (3 dim array) that contains all positive entris from both masks
numberToLooFor - number we will analyze whether is the same between two sets
cuda arrays holding just single value wit atomically reduced result
,fn,fp
,minxRes,maxxRes
,minyRes,maxyRes
,minZres,maxZres
dataBdim - dimensions of data block
metaDataDims - dimensions of the metadata
loopXMeta,loopYZMeta- indicates how many times we need to iterate over the metadata
inBlockLoopX,inBlockLoopY,inBlockLoopZ - indicates how many times we need to iterate over the data block using our size of thread block
                                          basically data block size will be established by the thread block size of main kernel  
"""
# function getBoolCubeKernel(goldGPU
#         ,segmGPU
#         ,numberToLooFor::T
#         ,reducedGoldA
#         ,reducedSegmA
#         ,reducedGoldB
#         ,reducedSegmB
#         ,fn
#         ,fp
#         ,minxRes
#         ,maxxRes
#         ,minyRes
#         ,maxyRes
#         ,minzRes
#         ,maxzRes
#         ,dataBdim
#         ,metaData
#         ,metaDataDims
#         ,mainArrDims
#         ,loopXMeta,loopYZMeta
#         ,inBlockLoopX,inBlockLoopY,inBlockLoopZ
# ) where T

@kernel function getBoolCube_kernel(goldGPU
        ,segmGPU
        ,numberToLooFor
        ,reducedGoldA
        ,reducedSegmA
        ,fn
        ,fp
        ,minxRes
        ,maxxRes
        ,minyRes
        ,maxyRes
        ,minzRes
        ,maxzRes
        ,dataBdim
        ,metaData
        ,metaDataDims
        ,mainArrDims
        ,loopMeta
        ,metaDataLength
        ,inBlockLoopX
        ,inBlockLoopY
        ,inBlockLoopZ
)
    @localAllocations()
    #we need nested x,y,z iterations so we will iterate over the matadata and on its basis over the  data in the main arrays 
    #first loop over the metadata 


    HFUtils.@iter3dOuter(metaDataDims,loopMeta,metaDataLength,
         begin
        # inner loop is over the data indicated by metadata
        @iterDataBlockZdeepest(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta
                         ,begin 
                                boolGold=goldGPU[x,y,z]==numberToLooFor
                                boolSegm=segmGPU[x,y,z]==numberToLooFor  
                  
                                
                                #we set all bits so we do not need to reset 
                                @setBitTo(offsetIter,zpos,boolGold)

                                # @setBitTo((shmemblockData[xpos,ypos]),zpos,boolGold)
                                #we need to also collect data about how many fp and fn we have in main part and borders
                                #important in case of corners we will first analyze z and y dims and z dim on last resort only !
  
                                # #in case some is positive we can go futher with looking for max,min in dims and add to the new reduced boolean arrays waht we are intrested in  
                                if(boolGold  || boolSegm)  
                                        anyPositive=true
                                        if((boolGold  xor boolSegm))
                                            @uploadLocalfpFNCounters()
                                            locFps+=boolSegm
                                            locFns+=boolGold
                                        end# if (boolGold  ⊻ boolSegm)
                                    #now                        

                                end#if boolGold  || boolSegm
                            end#ex
                          #additional after X
                            ,begin
                        # CUDA.@cuprint "xpos $(xpos) ypos $(ypos) loopXdim $(inBlockLoopX) xdim $(xdim) \n "    
                        #here we iterated over all z dimension so shmemblockData[xpos,ypos] is ready to be uploaded to global memory
                        if(offsetIter>0)            
                           @inbounds reducedGoldA[x,y,(zMeta+1)]=offsetIter
                        end
                        end)                


                  # now we are just after we iterated over a single data block - we need to save data about border data blocks 
                  # anyPositive = sync_threads_or(anyPositive) 
                  # Manual reduction for anyPositive using shared memory
                  shmemAnyPos = @localmem(Bool, 1)
                  if @index(Local, Linear) == 1
                      shmemAnyPos[1] = false
                  end
                  @synchronize
                  if anyPositive
                      shmemAnyPos[1] = true
                  end
                  @synchronize
                  anyPositive = shmemAnyPos[1]

                 @uploadMinMaxesToShmem()   


                 offsetIter=0       
                 #in order to reduce used shared memory we are setting values of output array separately
                 @iterDataBlockZdeepest(mainArrDims,dataBdim, inBlockLoopX,inBlockLoopY,inBlockLoopZ,xMeta,yMeta,zMeta
                 ,begin 
                    #if((segmGPU[x,y,z]==numberToLooFor)) @setBitTo1(offsetIter,zpos) end   
                      @setBitTo(offsetIter,zpos,(segmGPU[x,y,z]==numberToLooFor))
                       #we set all bits so we do not need to reset 
                    #@setBitTo(shmemblockData[xpos,ypos],zpos,(segmGPU[x,y,z]==numberToLooFor))
                end,begin 
                #here we iterated over all z dimension so shmemblockData[xpos,ypos] is ready to be uploaded to global memory
                if(offsetIter>0)  
                    @inbounds reducedSegmA[x,y,(zMeta+1)]=offsetIter
                end
                end)     
                @synchronize            

                    #we want to invoke this only once per data block
                    #save the data about number of fp and fn of this block and accumulate also this sum for global sum 
                    #doing all on first warp
                    @uploadDataToMetaData() 
                    # if(anyPositive)           
                    #     @ifXY 4 1  CUDA.@cuprint "xMeta $(xMeta+1)  ,yMeta $(yMeta+1),zMeta $(zMeta+1) \n"
                    # end    
                    #invoked after we gone through data block and now we save data into shared memory

                    @synchronize
                    #resetting
                    anyPositive= false  #reset                   
                    
                    @ifY 2 if(@index(Local, X)<15)
                        localQuesValues[@index(Local, X)]=0
                    end
               @synchronize

            end) #outer loop        
    #             #consider ceating tuple structure where we will have  number of outer tuples the same as z dim then inner tuples the same as y dim and most inner tuples will have only the entries that are fp or fn - this would make us forced to put results always in correct spots 


     @finalGlobalSet()


   return  
end
   end
   
"""
creates small memory footprint variables for getBoolCubeKernel
  return  minX,maxX,minY,maxY,minZ,maxZ,fn,fp
"""
function getSmallForBoolKernel(backend)
    minX = KernelAbstractions.allocate(backend, Float32, 1)
    maxX = KernelAbstractions.allocate(backend, Float32, 1)
    minY = KernelAbstractions.allocate(backend, Float32, 1)
    maxY = KernelAbstractions.allocate(backend, Float32, 1)
    minZ = KernelAbstractions.allocate(backend, Float32, 1)
    maxZ = KernelAbstractions.allocate(backend, Float32, 1)
    fn = KernelAbstractions.allocate(backend, UInt32, 1)
    fp = KernelAbstractions.allocate(backend, UInt32, 1)

    # Initialize
    KernelAbstractions.fill!(minX, 1110.0f0)
    KernelAbstractions.fill!(maxX, 0.0f0)
    KernelAbstractions.fill!(minY, 1110.0f0)
    KernelAbstractions.fill!(maxY, 0.0f0)
    KernelAbstractions.fill!(minZ, 1110.0f0)
    KernelAbstractions.fill!(maxZ, 0.0f0)
    KernelAbstractions.fill!(fn, 0)
    KernelAbstractions.fill!(fp, 0)

    return (minX, maxX, minY, maxY, minZ, maxZ, fn, fp)
end    


"""
creates large memory footprint variables for getBoolCubeKernel
    return reducedGoldA,reducedSegmA
"""
function getLargeForBoolKernel(backend, mainArrDims, dataBdim)
    # this is in order to be sure that array is divisible by data block so we reduce necessity of boundary checks
    xDim = cld(mainArrDims[1], dataBdim[1]) * dataBdim[1]
    yDim = cld(mainArrDims[2], dataBdim[2]) * dataBdim[2]
    zDim = cld(mainArrDims[3], dataBdim[3])
    newDims = (xDim, yDim, zDim)
    
    reducedGoldA = KernelAbstractions.allocate(backend, UInt32, newDims)
    reducedSegmA = KernelAbstractions.allocate(backend, UInt32, newDims)
    
    KernelAbstractions.fill!(reducedGoldA, 0)
    KernelAbstractions.fill!(reducedSegmA, 0)

    return (reducedGoldA, reducedSegmA)
end

# """
# iterating over shmemblockData
# """
# macro planeIter(loopXinPlane,loopYinPlane,maxXdim, maxYdim,ex)
#     mainExp = generalizedItermultiDim(
#     arrDims=:()
#     ,loopXdim=loopXinPlane
#     ,loopYdim=loopYinPlane
#     ,yCheck = :(y <=$maxYdim)
#     ,xCheck = :(x <=$maxXdim )
#     ,is3d = false
#     , ex = ex)
#       return esc(:( $mainExp))
# end


end#module


