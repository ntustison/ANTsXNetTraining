batchGenerator <- function( batchSize = 32L,
                            imageSize = c( 64, 64, 64 ),
                            template = NULL,
                            images = NULL,
                            brainMasks = NULL,
                            doRandomContralateralFlips = TRUE,
                            doDataAugmentation = TRUE
 )
{                                     

  if( is.null( template ) )
    {
    stop( "No reference template specified." )
    }
  if( is.null( images ) )
    {
    stop( "Input images must be specified." )
    }
  if( is.null( brainMasks ) )
    {
    stop( "Input masks must be specified." )
    }

  currentPassCount <- 0L

  function()
    {
    # Shuffle the data after each complete pass

    if( ( currentPassCount + batchSize ) >= length( images ) )
      {
      # shuffle the source data
      sampleIndices <- sample( length( images ) )
      images <- images[sampleIndices]
      brainMasks <- brainMasks[sampleIndices]

      currentPassCount <- 0L
      }

    batchIndices <- currentPassCount + 1L:batchSize

    batchImages <- images[batchIndices]
    batchBrainMasks <- brainMasks[batchIndices]

    X <- array( data = 0, dim = c( batchSize, imageSize, 2 ) )    
    Y <- array( data = 0L, dim = c( batchSize, imageSize ) )

    currentPassCount <<- currentPassCount + batchSize

    # pb <- txtProgressBar( min = 0, max = batchSize, style = 3 )

    i <- 1
    while( i <= batchSize )
      {
      # setTxtProgressBar( pb, i )

      t1 <- NULL 
      t2 <- NULL 
      batchT1 <- antsImageRead( batchImages[[i]][1] )
      batchT2 <- antsImageRead( batchImages[[i]][2] )
      batchBrainMask <- thresholdImage( antsImageRead( batchBrainMasks[i] ), 0.5, 1.0, 1, 0 )

      if( doRandomContralateralFlips && sample( c( TRUE, FALSE ) ) )
        {
        t1 <- reflectImage( batchT1, axis = 0, verbose = FALSE )  
        t2 <- reflectImage( batchT2, axis = 0, verbose = FALSE )  
        brainMask <- reflectImage( batchBrainMask, axis = 0, verbose = FALSE )
        } else {
        t1 <- batchT1
        t2 <- batchT2
        brainMask <- batchBrainMask
        }

      centerOfMassTemplate <- getCenterOfMass( template )
      centerOfMassImage <- getCenterOfMass( brainMask )
      xfrm <- createAntsrTransform( type = "Euler3DTransform",
        center = centerOfMassTemplate,
        translation = centerOfMassImage - centerOfMassTemplate )
      warpedT1 <- applyAntsrTransformToImage( xfrm, t1, template )
      warpedT2 <- applyAntsrTransformToImage( xfrm, t2, template )
      warpedBrainMask <- thresholdImage( applyAntsrTransformToImage( xfrm, brainMask, template ), 0.5, 1.0, 1, 0 )

      if( doDataAugmentation == TRUE )
        {
        dataAugmentation <- 
          randomlyTransformImageData( template, 
          list( list( warpedT1, warpedT2 ) ),
          list( warpedBrainMask ),
          numberOfSimulations = 1, 
          transformType = 'affineAndDeformation', 
          sdAffine = 0.1,
          deformationTransformType = "bspline",
          numberOfRandomPoints = 1000,
          sdNoise = 2.0,
          numberOfFittingLevels = 4,
          meshSize = 1,
          sdSmoothing = 4.0,
          inputImageInterpolator = 'linear',
          segmentationImageInterpolator = 'nearestNeighbor' )

        simulatedBrainMask <- thresholdImage( dataAugmentation$simulatedSegmentationImages[[1]], 0.5, 1.0, 1, 0 )
        simulatedT1 <- dataAugmentation$simulatedImages[[1]][[1]]
        simulatedT2 <- dataAugmentation$simulatedImages[[1]][[2]]
        simulatedT1Array <- as.array( simulatedT1 )
        simulatedT1Array <- ( simulatedT1Array - mean( simulatedT1Array ) ) / sd( simulatedT1Array )
        simulatedT2Array <- as.array( simulatedT2 )
        simulatedT2Array <- ( simulatedT2Array - mean( simulatedT2Array ) ) / sd( simulatedT2Array )

        # antsImageWrite( simulatedBrainMask, paste0( "./TempData/simBrainMask", i, ".nii.gz" ) )

        X[i,,,,1] <- simulatedT1Array
        X[i,,,,2] <- simulatedT2Array
        Y[i,,,] <- as.integer( as.array( simulatedBrainMask ) )

        i <- i + 1  
        } else {
        warpedT1Array <- as.array( warpedT1 )
        warpedT1Array <- ( warpedT1Array - mean( warpedT1Array ) ) / sd( warpedT1Array )        
        warpedT2Array <- as.array( warpedT2 )
        warpedT2Array <- ( warpedT2Array - mean( warpedT2Array ) ) / sd( warpedT2Array )        

        # antsImageWrite( warpedBrainMask, paste0( "./TempData/wBrainMask", i, ".nii.gz" ) )

        X[i,,,,1] <- warpedT1Array
        X[i,,,,2] <- warpedT2Array
        Y[i,,,] <- as.integer( as.array( warpedBrainMask ) )

        i <- i + 1  
        }
      }
    # stop( "Done testing.")  
    # cat( "\n" )

    encodedY <- encodeUnet( Y, c( 0, 1 ) )

    return( list( X, encodedY ) )
    }
}
