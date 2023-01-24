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

    X <- array( data = 0, dim = c( batchSize, imageSize, 1 ) )    
    Y <- array( data = 0L, dim = c( batchSize, imageSize ) )

    currentPassCount <<- currentPassCount + batchSize

    # pb <- txtProgressBar( min = 0, max = batchSize, style = 3 )
    
    i <- 1
    while( i <= batchSize )
      {
      # setTxtProgressBar( pb, i )

      image <- NULL 
      batchImage <- antsImageRead( batchImages[i] )
      batchBrainMask <- thresholdImage( antsImageRead( batchBrainMasks[i] ), 0.5, 1.0, 1, 0 )
      if( doRandomContralateralFlips && sample( c( TRUE, FALSE ) ) )
        {
        image <- reflectImage( batchImage, axis = 0, verbose = FALSE )  
        brainMask <- reflectImage( batchBrainMask, axis = 0, verbose = FALSE )
        } else {
        image <- batchImage  
        brainMask <- batchBrainMask
        }
      
      centerOfMassTemplate <- getCenterOfMass( template )
      centerOfMassImage <- getCenterOfMass( brainMask )
      xfrm <- createAntsrTransform( type = "Euler3DTransform",
        center = centerOfMassTemplate,
        translation = centerOfMassImage - centerOfMassTemplate )
      warpedImage <- applyAntsrTransformToImage( xfrm, image, template )
      warpedBrainMask <- thresholdImage( applyAntsrTransformToImage( xfrm, brainMask, template ), 0.5, 1.0, 1, 0 )
      
      if( doDataAugmentation == TRUE )
        {
        dataAugmentation <- 
          randomlyTransformImageData( template, 
          list( list( warpedImage ) ),
          list( warpedBrainMask ),
          numberOfSimulations = 1, 
          transformType = 'affineAndDeformation', 
          sdAffine = 0.01,
          deformationTransformType = "bspline",
          numberOfRandomPoints = 1000,
          sdNoise = 2.0,
          numberOfFittingLevels = 4,
          meshSize = 1,
          sdSmoothing = 4.0,
          inputImageInterpolator = 'linear',
          segmentationImageInterpolator = 'nearestNeighbor' )

        simulatedBrainMask <- thresholdImage( dataAugmentation$simulatedSegmentationImages[[1]], 0.5, 1.0, 1, 0 )
        simulatedImage <- dataAugmentation$simulatedImages[[1]][[1]]
        simulatedArray <- as.array( simulatedImage )
        simulatedArray <- ( simulatedArray - mean( simulatedArray ) ) / sd( simulatedArray )

        # antsImageWrite( simulatedImage, paste0( "./TempData/simBrainImage", i, ".nii.gz" ) )
        # antsImageWrite( simulatedBrainMask, paste0( "./TempData/simBrainMask", i, ".nii.gz" ) )

        X[i,,,,1] <- simulatedArray
        Y[i,,,] <- as.integer( as.array( simulatedBrainMask ) )

        i <- i + 1  
        } else {
        warpedArray <- as.array( warpedImage )
        warpedArray <- ( warpedArray - mean( warpedArray ) ) / sd( warpedArray )        

        # antsImageWrite( warpedImage, paste0( "./TempData/wBrainImage", i, ".nii.gz" ) )
        # antsImageWrite( warpedBrainMask, paste0( "./TempData/wBrainMask", i, ".nii.gz" ) )

        X[i,,,,1] <- warpedArray
        Y[i,,,] <- as.integer( as.array( warpedBrainMask ) )

        i <- i + 1  
        }
      }
#    stop( "Done testing.")  
    cat( "\n" )

    encodedY <- encodeUnet( Y, c( 0, 1 ) )
     
    return( list( X, encodedY, list( NULL ) ) )
    }
}
