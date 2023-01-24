batchGenerator <- function( batchSize = 32L,
                            imageSize = c( 64, 64 ),
                            images = NULL,
                            segmentations = NULL,
                            labels = NULL,
                            numberOfSlicesPerImage = 5,
                            doRandomContralateralFlips = TRUE,
                            doHistogramIntensityWarping = TRUE,
                            doAddNoise = TRUE,
                            doDataAugmentation = TRUE
 )
{

  if( is.null( images ) )
    {
    stop( "Input images must be specified." )
    }
  if( is.null( segmentations ) )
    {
    stop( "Input segmentations must be specified." )
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
      segmentations <- segmentations[sampleIndices]

      currentPassCount <- 0L
      }

    batchIndices <- currentPassCount + 1L:batchSize

    batchImages <- images[batchIndices]
    batchsegmentations <- segmentations[batchIndices]

    X <- array( data = 0, dim = c( batchSize, imageSize, 2 ) )
    Y <- array( data = 0L, dim = c( batchSize, imageSize ) )

    currentPassCount <<- currentPassCount + batchSize

#    pb <- txtProgressBar( min = 0, max = batchSize, style = 3 )

    i <- 1
    while( i <= batchSize )
      {
#      setTxtProgressBar( pb, i )

      image <- NULL
      segmentation <- NULL

      batchImage <- antsImageRead( batchImages[i] )
      # batchsegmentation <- thresholdImage( antsImageRead( batchsegmentations[i] ), 0, 0, 0, 1 )
      batchsegmentation <- antsImageRead( batchsegmentations[i] )

      if( doRandomContralateralFlips && sample( c( TRUE, FALSE ), 1 ) )
        {
        imageA <- as.array( batchImage )[dim(batchImage)[1]:1,,]
        segmentationA <- as.array( batchsegmentation )[dim(batchsegmentation)[1]:1,,]

      	image <- as.antsImage( imageA, reference = batchImage )
        segmentation <- as.antsImage( segmentationA, reference = batchsegmentation )
        } else {
      	image <- batchImage
        segmentation <- batchsegmentation
        }

      if( doHistogramIntensityWarping && sample( c( TRUE, FALSE ), 1 ) )
        {
	      breakPoints <- c( 0.2, 0.4, 0.6, 0.8 )
        displacements <- abs( rnorm( length( breakPoints ), 0, 0.175 ) )
        if( sample( c( TRUE, FALSE ), 1 ) )
          {
          displacements <- displacements * -1
          }
        image <- histogramWarpImageIntensities( image, breakPoints = breakPoints,
					        clampEndPoints = c( TRUE, TRUE ),
					        displacements = displacements )
        }

      imageX <- image
      segmentationX <- segmentation
      if( doDataAugmentation == TRUE )
        {
        dataAugmentation <-
          randomlyTransformImageData( image,
          list( list( image ) ),
          list( segmentation ),
          numberOfSimulations = 1,
          transformType = 'affine',
          sdAffine = 0.01,
          deformationTransformType = "bspline",
          numberOfRandomPoints = 1000,
          sdNoise = 2.0,
          numberOfFittingLevels = 4,
          meshSize = 1,
          sdSmoothing = 4.0,
          inputImageInterpolator = 'linear',
          segmentationImageInterpolator = 'nearestNeighbor' )

        imageX <- dataAugmentation$simulatedImages[[1]][[1]]
        segmentationX <- dataAugmentation$simulatedSegmentationImages[[1]]
        }

      imageX <- ( imageX - mean( imageX ) ) / sd( imageX )
      if( doAddNoise && sample( c( TRUE, FALSE ), 1 ) )
        {
        imageX <- addNoiseToImage( imageX, noiseModel = "additivegaussian", c( 0, runif( 1, 0, 0.05 ) ) )
        imageX <- ( imageX - mean( imageX ) ) / sd( imageX )
        }

      segmentationArrayX <- as.array( segmentationX )
      imageArrayX <- as.array( imageX )

      maskX <- thresholdImage( segmentationX, 0, 0, 0, 1 )

      geoms <- labelGeometryMeasures( maskX )
      if( length( geoms$Label ) == 0 )
        {
        next
        }

      whichDimensionMaxSpacing <- which.max( antsGetSpacing( imageX ) )[1]
      if( whichDimensionMaxSpacing == 1 )
        {
        lowerSlice <- geoms$BoundingBoxLower_x[1] + 1
        upperSlice <- geoms$BoundingBoxUpper_x[1]
        } else if( whichDimensionMaxSpacing == 2 ) {
        lowerSlice <- geoms$BoundingBoxLower_y[1] + 1
        upperSlice <- geoms$BoundingBoxUpper_y[1]
        } else {
        lowerSlice <- geoms$BoundingBoxLower_z[1] + 1
        upperSlice <- geoms$BoundingBoxUpper_z[1]
        }
      if( lowerSlice >= upperSlice )
        {
        next
        }

      whichRandomSlices <- sample( seq.int( from = lowerSlice, to = upperSlice, by = 1 ),
        min( numberOfSlicesPerImage, upperSlice - lowerSlice + 1 ) )
      for( j in seq.int( length( whichRandomSlices ) ) )
        {
        whichSlice <- whichRandomSlices[j]

        imageSlice <- NULL
        segmentationSlice <- NULL
        if( whichDimensionMaxSpacing == 1 )
          {
          imageSlice <- as.antsImage( drop( imageArrayX[whichSlice,,] ) )
          segmentationSlice <- as.antsImage( drop( segmentationArrayX[whichSlice,,] ) )
          } else if( whichDimensionMaxSpacing == 2 ) {
          imageSlice <- as.antsImage( drop( imageArrayX[,whichSlice,] ) )
          segmentationSlice <- as.antsImage( drop( segmentationArrayX[,whichSlice,] ) )
          } else {
          imageSlice <- as.antsImage( drop( imageArrayX[,,whichSlice] ) )
          segmentationSlice <- as.antsImage( drop( segmentationArrayX[,,whichSlice] ) )
          }

        imageSlice <- padOrCropImageToSize( imageSlice, imageSize )
        segmentationSlice <- padOrCropImageToSize( segmentationSlice, imageSize )
        maskSlice <- smoothImage( thresholdImage( segmentationSlice, 0, 0, 0, 1 ), 1.0 )

        X[i,,,1] <- as.array( imageSlice )
        X[i,,,2] <- as.array( maskSlice )
        Y[i,,] <- as.array( segmentationSlice )

        i <- i + 1
        if( i >= batchSize )
          {
          break
          }
        }
      }
 #   cat( "\n" )
    encodedY <- encodeUnet( Y, labels )

    return( list( X, encodedY ) )
    }
}
