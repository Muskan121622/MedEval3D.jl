module MainLoopKernel
using KernelAbstractions, Atomix, CUDA, Logging,..CUDAGpuUtils, ..ResultListUtils,..WorkQueueUtils,..ScanForDuplicates, Logging,StaticArrays, ..IterationUtils, ..ReductionUtils, ..CUDAAtomicUtils,..MetaDataUtils
using ..BitWiseUtils,..MetadataAnalyzePass, ..ScanForDuplicates, ..ProcessMainDataVerB
export mainKernelLoad,@mainLoopKernelAllocations,getSmallGPUForHousedorff,getBigGPUForHousedorffAfterBoolKernel,@loadDataAtTheBegOfDilatationStep,@mainLoopKernel, @iterateOverWorkQueue,@mainLoop,@mainLoopKernelAllocations,@clearBeforeNextDilatation


"""
clearing data in order to enable their reuse in next iteration
clearIterResShmemLoop - given we trat res shmem as one dimensional array with 32 entries per iteration how many times we need to loop to cover all
clearIterSourceShmemLoop - the same as above but for source shmem
resShmemTotalLength, sourceShmemTotalLength - total (treating as 1 dimensional array) length of the  sourse or result shared memory
"""
macro clearBeforeNextDilatation( clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength,dataBdim)
    return esc(quote
    isMaskFull=true

    @ifY 1 if(@index(Local, X)<15) areToBeValidated[@index(Local, X)]=false end 
    @ifY 2 if(@index(Local, X)<8) isAnythingInPadding[@index(Local, X)]=false end 
   

end)#quote
end#clearBeforeNextDilatation




"""
allocates memory in the kernel some register and shared memory  (no allocations in global memory here from kernel)
dataBdim - are dimensions of data block - each data block has just one row in the metadata ...
"""
macro mainLoopKernelAllocations(dataBdim)
    return esc(quote
        # shmemblockData = @localmem(UInt32, (32, 32, 2))
        # holding data about result  2)bottom, 3)left 4)right , 5)anterior, 6)posterior , 7) top,  paddings
        shmemPaddings = @localmem(Bool, (32, 32, 7))

        # for storing sums for reductions
        shmemSum = @localmem(UInt32, (36, 16)) # we need this additional spots
        # used to load from metadata information are ques to be validated 
        areToBeValidated = @localmem(Bool, 14) 
        # used to mark wheather there is any true in paddings
        isAnythingInPadding = @localmem(Bool, 7)
        # used to accumulate counts from of fp's and fn's already covered in dilatation steps
        alreadyCoveredInQueues = @localmem(UInt32, 14)
         
        # boolean usefull in the iterating over private part of work queue 
        isAnyBiggerThanZero = @localmem(Bool, 1)
        # loading data to shared memory from global about is it even or odd pass
        iterationNumberShmem = @localmem(UInt16, 1)
        # shared memory variables needed to marks wheather we are already finished with  any dilatation step
        goldToBeDilatated = @localmem(Bool, 1)
        segmToBeDilatated = @localmem(Bool, 1)
        # true when we have more than 0 blocks to analyze in next iteration
        isEvenPass = @localmem(Bool, 1) 
        # keeping information  weather we have not odd or even pass
        workCounterBiggerThan0 = @localmem(Bool, 1) 
        # holding values of  work queue counter ghlobal 
        workCounterInshmem = @localmem(UInt16, 1)
        # workCounter of this block - thread local
        workCounterLocalInShmem = @localmem(UInt16, 1)
        # helper variable used to add data to global counter
        workCounterHelper = @localmem(UInt16, 1)
        positionInMainWorkQueaue = @localmem(UInt16, 1)

        @ifXY 1 1 iterationNumberShmem[1] = 0
        @ifXY 2 1 isAnyBiggerThanZero[1] = 0
        @ifXY 3 1 workCounterInshmem[1] = 0
        @ifXY 7 1 goldToBeDilatated[1] = 0
        @ifXY 8 1 segmToBeDilatated[1] = 0
        @ifXY 9 1 workCounterBiggerThan0[1] = 0
        @ifXY 19 1 isEvenPass[1] = true
        
        @synchronize
    end)
end




"""
main loop logic
"""
macro mainLoop()
    return esc(quote
    #we already established work queue for first pass
    @clearBeforeNextDilatation( clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength,dataBdim)

    @loadDataAtTheBegOfDilatationStep()

    @synchronize
    
    #now we should have all data needed to analyze padding from previous dilatation
    @iterateOverWorkQueue(begin     
        @executeIterPadding(dilatationArrs[shmemSum[shmemIndex*4+4]+1]
        ,referenceArrs[shmemSum[shmemIndex*4+4]+1]
        ,shmemSum[shmemIndex*4+1]#xMeta
        ,shmemSum[shmemIndex*4+2]#yMeta
        ,shmemSum[shmemIndex*4+3]#zMeta
        ,shmemSum[shmemIndex*4+4]#isGold
        ,iterationNumberShmem[1]#iterNumb
        )

    end ) 
    # sync_grid(grid_handle) # KA doesn't support grid sync; need to handle via separate kernel launches if needed

    #we get dilatation from block its padding will be analyzed later
    @iterateOverWorkQueue(begin 
        @executeDataIter(mainArrDims 
        ,dilatationArrs[shmemSum[shmemIndex*4+4]+1]
        ,referenceArrs[shmemSum[shmemIndex*4+4]+1]
        ,shmemSum[shmemIndex*4+1]#xMeta
        ,shmemSum[shmemIndex*4+2]#yMeta
        ,shmemSum[shmemIndex*4+3]#zMeta
        ,shmemSum[shmemIndex*4+4]#isGold
        ,iterationNumberShmem[1]#iterNumb
    )
    end ) 
    # sync_grid(grid_handle)
    #every time we are overwriting work queaue
    @ifXY 1 1 workQueueCounter[1]=0
    
    # sync_grid(grid_handle)

     MetadataAnalyzePass.@setMEtaDataOtherPasses(locArr,offsetIter,iterationNumberShmem[1])

  

end) #quote
end#mainLoop



"""
iterating over elements in work queaue  in order to make it work we need  to 
    know how many elements are in the work queue - we need workQueueCounter
    workQueaue - matrix with work queue data  
    goldToBeDilatated, segmToBeDilatated - shared memory values needed to establish weather we finished dilatations in given pass
    shmemSum will be used to store loaded data from workqueue 
    shmemSumLengthMaxDiv4 - number indicating linear length of shmemSum but reduced such that it will be divisible by 4
    ex - actions invoked on the data block when its xMeta,yMeta,zMeta and is gold pass informations are already known
    """
macro iterateOverWorkQueue(ex )
   return esc(quote
    #first part we load data from work queue to shmem sum 
    # we will treat shmemSum as 1 dimensional array and write data from work queue
    #mod 1 - xMeta, mod 2 - uMeta, mod 3 - zMeta mod 4 - isGoldPass
    #first we need to  establish how many items in work queue will be analyzed by this block 
   numbOfDataBlockPerThreadBlock = cld(workQueueCounter[1],@numgroups()[1] )

    #we need to stuck all of the blocks data into shared memory 4 entries for each block
    @unroll for outerIter in 0: fld((numbOfDataBlockPerThreadBlock*4),shmemSumLengthMaxDiv4)
        workQuueueLinearIndexOffset = ((((numbOfDataBlockPerThreadBlock)*4 ))*(@group_id(X)-1))+ (outerIter*shmemSumLengthMaxDiv4)
        # now we load all needed data into shared memory
        @iterateLinearly cld(shmemSumLengthMaxDiv4,@groupsize()[1]*@groupsize()[2]) shmemSumLengthMaxDiv4 begin
            #checking if we are in range
            if(i<=shmemSumLengthMaxDiv4 )
                if( ((outerIter*shmemSumLengthMaxDiv4)+i)<=((numbOfDataBlockPerThreadBlock*4)) && (workQuueueLinearIndexOffset +i)<=(workQueueCounter[1]*4)  )
                    shmemSum[i] = workQueue[(workQuueueLinearIndexOffset +i)]
                else
                    shmemSum[i] =0      
                end
            end
            
        end
        @synchronize
    #at this point we had pushed into shared memory as much data as we can fit so we need to start using it         
    #second part proper iteration by definition no rounding needed here 
    #also we do not make here any attempt of parallelization as this will be done inside the expression we just provide actual metadata for dilatation step
        for shmemIndex in 0:(fld(shmemSumLengthMaxDiv4,4)-1)
                if( ((shmemIndex+1)*4 <=shmemSumLengthMaxDiv4 ) && shmemSum[shmemIndex*4+1]>0  ) #shmemSum[shmemIndex*4+1]>0
                # checking is there any point in futher dilatations of this block
                if((shmemSum[shmemIndex*4+4]==1 && goldToBeDilatated[1]) || (shmemSum[shmemIndex*4+4]==0 && segmToBeDilatated[1]) )
                    #finally all ready for dilatation step to be executed on this particular block
                    #this one below should be removed after problem is diagnosed
                    if(shmemSum[shmemIndex*4+1]< mainArrDims[1]  && shmemSum[shmemIndex*4+2]< mainArrDims[2]&& shmemSum[shmemIndex*4+3]< mainArrDims[2] && shmemSum[shmemIndex*4+4]< 2) 
                        $ex
                    end    
                end    
            end#if in range
        end# main functional loop for dilatation and validation  
    end#outer for 
    end)#quote 
end    


"""
main kernel managing 
"""
@kernel function mainLoop_kernel(referenceArrs, dilatationArrs, mainArrDims, dataBdim
    , numberToLooFor, metaDataDims, metaData, iterThrougWarNumb, robustnessPercent
    , shmemSumLengthMaxDiv4, globalFpResOffsetCounter, globalFnResOffsetCounter
    , globalIterationNumber, globalCurrentFnCount, globalCurrentFpCount
    , globalIterationNumb, workQueue, workQueueCounter
    , loopAXFixed, loopBXfixed, loopAYFixed, loopBYfixed, loopAZFixed, loopBZfixed
    , loopdataDimMainX, loopdataDimMainY, loopdataDimMainZ, inBlockLoopX, inBlockLoopY
    , inBlockLoopZ, metaDataLength, loopMeta, loopWarpMeta, clearIterResShmemLoop
    , clearIterSourceShmemLoop, resShmemTotalLength, sourceShmemTotalLength, fn, fp, resList, inBlockLoopXZIterWithPadding, paddingStore, shmemblockDataLenght, shmemblockDataLoop)
    
    @mainLoopKernelAllocations(dataBdim)

    @clearBeforeNextDilatation(clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength,dataBdim)    
    
    MetadataAnalyzePass.@analyzeMetadataFirstPass()

    # sync_grid(grid_handle)   


    @loadDataAtTheBegOfDilatationStep()

    # sync_grid(grid_handle)   
    for i in 1:11
        @mainLoop()
    end     
    @ifXY 1 1 globalIterationNumb[1]=iterationNumberShmem[1]     
end

macro mainLoopKernel()
  return esc(quote
    # This macro is now deprecated in favor of mainLoop_kernel
end)#quote
end





# """
# after dilatation prepare
# """
# macro prepareForNextDilation()
#     return esc(quote
#     #we update metadata and prepare work queue
#     setMEtaDataOtherPasses()
#     #we clear  and add negation to !isOddPassShmem becouse we want to set the previously updated counter to 0 
#     @clearBeforeNextDilatation( clearIterResShmemLoop,clearIterSourceShmemLoop,resShmemTotalLength, sourceShmemTotalLength,dataBdim)

#     @loadDataAtTheBegOfDilatationStep()
# end)#quote
# end    #prepareForNextDilation



        
"""
loads data at the begining of each dilatation step
    we need to set some variables in shared memory ro initial values
    fp, fn  
"""
macro loadDataAtTheBegOfDilatationStep(  )

    return esc(quote
    #so we know that becouse of sync grid we will have evrywhere the same  iterationNumberShmem and positionInMainWorkQueaue
    
    @ifXY 1 1 iterationNumberShmem[1]+=1
    #@ifXY 2 2 positionInMainWorkQueaue[1]=0 

    @ifXY 2 1 begin
        workCounterInshmem[1]= workQueueCounter[1] 
        workCounterBiggerThan0[1]= (workCounterInshmem[1]>0)
                    end 
    #we do corection for robustness so we can ignore some of the most distant points - this will reduce the influence of outliers                
    @ifXY 3 1 goldToBeDilatated[1]=(globalCurrentFpCount[1] <= ceil(fp[1]*robustnessPercent))
    @ifXY 4 1 segmToBeDilatated[1]=(globalCurrentFnCount[1] <= ceil(fn[1]*robustnessPercent))
        end)#quote
    end



"""
allocates memory for small GPU Arrays
"""
function getSmallGPUForHousedorff()
    globalFpResOffsetCounter= CUDA.zeros(UInt32,1)
    globalFnResOffsetCounter= CUDA.zeros(UInt32,1)
    workQueueCounter= CUDA.zeros(UInt32,1)
    globalIterationNumber = CUDA.zeros(UInt32,1)
    globalCurrentFnCount= CUDA.zeros(UInt32,1)
    globalCurrentFpCount= CUDA.zeros(UInt32,1)
    globalIterationNumb= CUDA.zeros(UInt32,1)

return (globalFpResOffsetCounter,globalFnResOffsetCounter,globalIterationNumber,globalCurrentFnCount,globalCurrentFpCount,globalIterationNumb )
end    

# """
# allocate memory for bigger arrays 
# """
# function getBigGPUForHousedorff()


# end 


"""
allocate after prepare bool kernel had finished execution
return metaData,reducedGoldA,reducedSegmA,reducedGoldB,reducedSegmB,resList,workQueue,,workQueuecounter
"""
function getBigGPUForHousedorffAfterBoolKernel(metaData,minxRes,maxxRes,minyRes,maxyRes,minzRes,maxzRes,fn,fp,reducedGoldA,reducedSegmA,dataBdim)
    ###we return only subset of boolean arrays that we are intrested in 
    # println( "zzz fp[1] $(fp[1])  fn[1] $(fn[1]) \n")
    # resList = allocateResultLists(fp[1],fn[1])
    resList = allocateResultLists(1000,1000)
    ###we need to return subset of metadata that we are intrested in 
    goldArr= reducedGoldA[Int64(minxRes[1]*dataBdim[1]):Int64(maxxRes[1]*dataBdim[1])
            ,Int64(minyRes[1]*dataBdim[2]):Int64(maxyRes[1]*dataBdim[2])
            ,Int64(minzRes[1]):Int64(maxzRes[1])
            ]
    segmArr = reducedSegmA[Int64(minxRes[1]*dataBdim[1]):Int64(maxxRes[1]*dataBdim[1])
                ,Int64(minyRes[1]*dataBdim[2]):Int64(maxyRes[1]*dataBdim[2])
                ,Int64(minzRes[1]):Int64(maxzRes[1])
                ]
    
    # CUDA.unsafe_free!(reducedGoldA)
    
    # CUDA.unsafe_free!(reducedSegmA)
    newMeta = metaData[Int64(minxRes[1]):Int64(maxxRes[1]),Int64(minyRes[1]):Int64(maxyRes[1]),Int64(minzRes[1]):Int64(maxzRes[1]),:   ]
    workQueue,workQueueCounter= WorkQueueUtils.allocateWorkQueue( max(length(newMeta),1) )
    metaSize = size(newMeta)



    #here we need to define data structure that will hold all data about paddings
    # it will be composed of 32x32 blocks of UInt8 where first 6 bits will represent paddings 
    # number of such blocks will be the same as innersingleDataBlockPass
    paddingStore = CUDA.zeros(UInt8, metaSize[1],metaSize[2],metaSize[3],32,32)
    
    
    return(newMeta
            ,goldArr  ,segmArr ,paddingStore
            ,resList,workQueue,workQueueCounter
    )
end 




end#MainLoopKernel



# """
# analyzing private part of work queue
# """
# macro privateWorkQueueAnalysis()
#     return esc(quote
 
#     loadOwnedWorkQueueIntoShmem(mainWorkQueue,mainQuesCounter,toIterWorkQueueShmem,positionInMainWorkQueaue ,numberOfThreadBlocks,tailParts)
#     #at this point if we have anything in the private part of the  work queue we will have it in toIterWorkQueueShmem, in case  we will find some 0 inside it means that queue is exhousted and we need to loo into tail
#     @unroll for i in UInt16(1):32# most outer loop is responsible for z dimension
#         if(toIterWorkQueueShmem[i,1]>0)
#             #we need to check also wheather given dilatation step is already finished - for example it is possible that it do not make sense to dilatate gold mask but still we need dilate other
#             if( (goldToBeDilatated[1]&&toIterWorkQueueShmem[i,4]==1) || (segmToBeDilatated[1]&&toIterWorkQueueShmem[i,4]==0)  )    
#                 @innersingleDataBlockPass(toIterWorkQueueShmem[i,4]==1,toIterWorkQueueShmem[i,1],toIterWorkQueueShmem[i,2] ,toIterWorkQueueShmem[i,3] )
#             end
#             sync_threads()
#         else    #at this point we got some 0 in the toIterWorkQueueShmem 
#             @ifXY 1 1  isAnyBiggerThanZero[]=false
#             sync_threads()
#         end#if    
#     end#for    
# end)#quote
# end    


# """
# analyzing tail part of work queue
# """
# macro analyzeTail()
#     return esc(quote

#     #below we access tail of working queue in a way that will be atomic
#     @ifXY 1 1 toIterWorkQueueShmem[1]= mainWorkQueueArr[isOddPassShmem[]+1,currentTailPosition[1]]
#     sync_threads()
#     #we need to check also wheather given dilatation step is already finished - for example it is possible that it do not make sense to dilatate gold mask but still we need dilate other
#     if( (goldToBeDilatated[1]&&toIterWorkQueueShmem[1,4]==1) || (segmToBeDilatated[1]&&toIterWorkQueueShmem[1,4]==0)  )    
#         @innersingleDataBlockPass(toIterWorkQueueShmem[1,4]==1,toIterWorkQueueShmem[1,1],toIterWorkQueueShmem[1,2] ,toIterWorkQueueShmem[1,3] )
#     end
#     loadDataNeededForTailAnalysisToShmem(currentTailPosition,tailCounter )
#     sync_threads()
# end) #quote
# end
