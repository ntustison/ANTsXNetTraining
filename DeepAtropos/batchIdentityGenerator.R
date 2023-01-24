batchGenerator <- function( batchSize = 32L,
                            imageSize = c( 64, 64, 64 ),
                            template = NULL,
                            images = NULL,
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

  currentPassCount <- 0L

  function()
    {
    # Shuffle the data after each complete pass

    if( ( currentPassCount + batchSize ) >= length( images ) )
      {
      # shuffle the source data
      sampleIndices <- sample( length( images ) )
      images <- images[sampleIndices]

      currentPassCount <- 0L
      }

    batchIndices <- currentPassCount + 1L:batchSize

    batchImages <- images[batchIndices]

    X <- array( data = 0, dim = c( batchSize, imageSize, 1 ) )    
    Y <- array( data = 0, dim = c( batchSize, imageSize, 1 ) )

    currentPassCount <<- currentPassCount + batchSize

    # pb <- txtProgressBar( min = 0, max = batchSize, style = 3 )
    
    i <- 1
    while( i <= batchSize )
      {
      # setTxtProgressBar( pb, i )

      image <- NULL 
      batchImage <- antsImageRead( batchImages[i] )
      if( doRandomContralateralFlips && sample( c( TRUE, FALSE ) ) )
        {
        image <- reflectImage( batchImage, axis = 0, verbose = FALSE )  
        } else {
        image <- batchImage  
        }
      
      warpedImage <- image
      
      if( doDataAugmentation == TRUE )
        {
        dataAugmentation <- 
          randomlyTransformImageData( template, 
          list( list( warpedImage ) ),
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

        X[i,,,,1] <- simulatedArray[13:172,15:206,1:160]
        Y[i,,,,1] <- simulatedArray[13:172,15:206,1:160]

        i <- i + 1  
        } else {
        warpedArray <- as.array( warpedImage )
        warpedArray <- ( warpedArray - mean( warpedArray ) ) / sd( warpedArray )        

        X[i,,,,1] <- warpedArray[13:172,15:206,1:160]
        Y[i,,,,1] <- warpedArray[13:172,15:206,1:160]

        i <- i + 1  
        }
      }
    # stop( "Done testing.")  
    # cat( "\n" )

    return( list( X, Y ) )
    }
}
