batchGenerator <- function( batchSize = 32L,
                            imageSize = c( 64, 64, 64 ),
                            numberOfBins = 40,
                            sigma = 1,
                            ageSpan = c( 42, 82 ),
                            template = NULL,
                            reflectionMatrix = NULL,
                            doDataAugmentation = FALSE,
                            doRegression = FALSE,
                            images = NULL,
                            ages = NULL
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
  if( is.null( ages ) )
    {
    stop( "No ages specified." )
    }

  doRandomContralateralFlips <- FALSE
  if( ! is.null( reflectionMatrix ) )
    {
    doRandomContralateralFlips <- TRUE  
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
      ages <- ages[sampleIndices]

      currentPassCount <- 0L
      }

    batchIndices <- currentPassCount + 1L:batchSize

    batchImages <- images[batchIndices]
    batchAges <- ages[batchIndices]

    X1 <- array( data = 0, dim = c( batchSize, imageSize, 1 ) ) 
    Y1 <- array( data = 0, dim = c( batchSize, numberOfBins ) )
    if( doRegression == TRUE )
      {
      Y1 <- array( data = 0, dim = c( batchSize ) )
      }  

    currentPassCount <<- currentPassCount + batchSize

    # pb <- txtProgressBar( min = 0, max = batchSize, style = 3 )
    for( i in seq_len( batchSize ) )
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
      
      if( doDataAugmentation == TRUE )
        {
        dataAugmentation <- 
          randomlyTransformImageData( template, 
          list( list( image ) ),
          numberOfSimulations = 1, 
          transformType = 'affine', 
          sdAffine = 0.02,
          deformationTransformType = "bspline",
          numberOfRandomPoints = 1000,
          sdNoise = 10.0,
          numberOfFittingLevels = 4,
          meshSize = 1,
          sdSmoothing = 4.0,
          inputImageInterpolator = 'linear',
          segmentationImageInterpolator = 'nearestNeighbor' )

        simulatedImage <- dataAugmentation$simulatedImages[[1]][[1]]
        simulatedArray <- as.array( simulatedImage ) / mean( simulatedImage )
        X1[i,,,,1] <- simulatedArray[11:170,13:204,11:170]
        } else {
        imageArray <- as.array( image ) / mean( image )
        X1[i,,,,1] <- imageArray[11:170,13:204,11:170]
        }
      # antsImageWrite( as.antsImage( drop( X1[i,,,,1] ) ), paste0( "./TempData/X1_", i, ".nii.gz" ) )
      # cat( paste0( batchImages[i], "\n" ), file = './TempData/imageFiles.txt', append = TRUE )

      if( doRegression == TRUE ) 
        {
        Y1[i] <- batchAges[i]  
        } else {
        ageDelta <- ( ageSpan[2] - ageSpan[1] ) / numberOfBins
        ageBins <- seq( from = ageSpan[1] + 0.5 * ageDelta, 
          to = ageSpan[2] - 0.5 * ageDelta, length.out = numberOfBins )
        y <- rep( 0, numberOfBins )  
        for( j in seq.int( numberOfBins ) )  
          {
          agem1 <- ageBins[j] - 0.5 * ageDelta
          agep1 <- ageBins[j] + 0.5 * ageDelta
          ageCdf = pnorm( c( agem1, agep1 ), mean = batchAges[i], sd = sigma )
          y[j] <- ageCdf[2] - ageCdf[1]
          }
        # y <- max( min( y, 1.0 ), 1.0e-7 )  
        Y1[i,] <- y
        # write.csv( drop( Y1 ), paste0( "./TempData/simBrainImage", i, ".csv" ), row.names = FALSE )
        }
      }
    # stop( "Done testing." )  
    # cat( "\n" )
    return( list( X1, Y1 ) )
    }
}
