batchGenerator <- function( batchSize = 32L,
                            imageSize = c( 64, 64, 64 ),
                            template = NULL,
                            images = NULL,
                            segmentationImages = NULL,
			    segmentationLabels = NULL,
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
  if( is.null( segmentationImages ) )
    {
    stop( "Input masks must be specified." )
    }
  if( is.null( segmentationLabels ) )
    {
    stop( "segmentationLabels must be specified." )
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
      segmentationImages <- segmentationImages[sampleIndices]

      currentPassCount <- 0L
      }

    batchIndices <- currentPassCount + 1L:batchSize

    batchImages <- images[batchIndices]
    batchSegmentationImages <- segmentationImages[batchIndices]

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
      batchSegmentationImage <- antsImageRead( batchSegmentationImages[i] )
      if( doRandomContralateralFlips && sample( c( TRUE, FALSE ) ) )
        {
        image <- reflectImage( batchImage, axis = 0, verbose = FALSE )  
        segmentationImage <- reflectImage( batchSegmentationImage, axis = 0, verbose = FALSE )
        } else {
        image <- batchImage  
        segmentationImage <- batchSegmentationImage
        }
      
      # centerOfMassTemplate <- getCenterOfMass( template )
      # centerOfMassImage <- getCenterOfMass( segmentationImage )
      # xfrm <- createAntsrTransform( type = "Euler3DTransform",
      #   center = centerOfMassTemplate,
      #   translation = centerOfMassImage - centerOfMassTemplate )
      # warpedImage <- applyAntsrTransformToImage( xfrm, image, template )
      # warpedMask <- thresholdImage( applyAntsrTransformToImage( xfrm, segmentationImage, template ), 0.5, 1.0, 1, 0 )

      warpedImage <- image
      warpedMask <- segmentationImage
      
      if( doDataAugmentation == TRUE )
        {
        dataAugmentation <- 
          randomlyTransformImageData( template, 
          list( list( warpedImage ) ),
          list( warpedMask ),
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

        simulatedImage <- dataAugmentation$simulatedImages[[1]][[1]]
        simulatedArray <- as.array( simulatedImage )
        simulatedArray <- ( simulatedArray - mean( simulatedArray ) ) / sd( simulatedArray )

        simulatedMask <- dataAugmentation$simulatedSegmentationImages[[1]]
        sumulatedMaskArray <- as.array( simulatedMask )


        # antsImageWrite( simulatedImage, paste0( "./TempData/simBrainImage", i, ".nii.gz" ) )
        # antsImageWrite( simulatedSegmentationImage, paste0( "./TempData/simSegmentationImage", i, ".nii.gz" ) )

        X[i,,,,1] <- simulatedArray[13:172,15:206,1:160]
        Y[i,,,] <- simulatedMaskArray[13:172,15:206,1:160]

        i <- i + 1  
        } else {
        warpedArray <- as.array( warpedImage )
        warpedArray <- ( warpedArray - mean( warpedArray ) ) / sd( warpedArray )        
        warpedMaskArray <- as.array( warpedMask )

        # antsImageWrite( warpedImage, paste0( "./TempData/wBrainImage", i, ".nii.gz" ) )
        # antsImageWrite( warpedSegmentationImage, paste0( "./TempData/wSegmentationImage", i, ".nii.gz" ) )

        X[i,,,,1] <- warpedArray[13:172,15:206,1:160]
        Y[i,,,] <- warpedMaskArray[13:172,15:206,1:160]

        i <- i + 1  
        }
      }
    # stop( "Done testing.")  
    # cat( "\n" )

    encodedY <- encodeUnet( Y, segmentationLabels )

    return( list( X, encodedY ) )
    }
}
