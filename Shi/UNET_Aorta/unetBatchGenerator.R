unetImageBatchGenerator <- function( batchSize = 32L,
                                     segmentationLabels = c( 0, 1 ),
                                     imageList = NULL,
                                     segmentationList = NULL )
{
  if( is.null( imageList ) )
    {
    stop( "Input images must be specified." )
    }
  if( is.null( segmentationList ) )
    {
    stop( "Input segmentation images must be specified." )
    }

  rotate90 <- function( matrix )
    {
    rotatedMatrix <- t( matrix[nrow( matrix ):1,, drop = FALSE] )
    return( rotatedMatrix )
    }

  flipVertically <- function( matrix )
    {
    tmp <- matrix
    flippedMatrix <- apply( tmp, 2, rev )
    return( flippedMatrix )
    }

  flipHorizontally <- function( matrix )
    {
    tmp <- matrix
    flippedMatrix <- apply( tmp, 1, rev )
    return( flippedMatrix )
    }

  currentPassCount <- 0L

  function()
    {
    # Shuffle the data after each complete pass

    if( ( currentPassCount + batchSize ) >= length( imageList ) )
      {
      currentPassCount <- 0L
      }

    batchIndices <- sample( seq_len( length( imageList ) ), batchSize, replace = TRUE )

    batchImages <- imageList[batchIndices]
    batchSegmentations <- segmentationList[batchIndices]

    channelSize <- 3
    imageSize <- c( 512, 512 )

    batchX <- array( data = 0, dim = c( batchSize, imageSize, channelSize ) )
    batchY <- array( data = 0, dim = c( batchSize, imageSize ) )

    currentPassCount <<- currentPassCount + batchSize

    # pb <- txtProgressBar( min = 0, max = batchSize, style = 3 )
    for( i in seq_len( batchSize ) )
      {
      segmentation <- antsImageRead( batchSegmentations[[i]] )
      segmentationArray <- drop( as.array( segmentation ) )
      imageArray <- as.array( load.image( batchImages[[i]] ) )[,,1,, drop = TRUE]

      # cat( "i: ", i, batchImages[[i]], "\n" )
      # cat( "s: ", i, batchSegmentations[[i]], "\n" )

      doVerticalFlip <- sample( c( TRUE, FALSE ), 1 )
      if( doVerticalFlip )
        {
        segmentationArray <- flipVertically( segmentationArray )
        for( d in seq_len( channelSize ) )
          {
          imageArray[,, d] <- flipVertically( imageArray[,, d] )
          }
        }

      doHorizontalFlip <- sample( c( TRUE, FALSE ), 1 )
      if( doHorizontalFlip )
        {
        segmentationArray <- flipHorizontally( segmentationArray )
        for( d in seq_len( channelSize ) )
          {
          imageArray[,, d] <- flipHorizontally( imageArray[,, d] )
          }
        }

      numberOfRotations <- sample( c( 0, 1, 2, 3 ), 1 )
      if( numberOfRotations > 0 )
        {
        for( i in seq_len( numberOfRotations ) )
          {
          segmentationArray <- rotate90( segmentationArray )
          for( d in seq_len( channelSize ) )
            {
            imageArray[,, d] <- rotate90( imageArray[,, d] )
            }
          }
        }

      batchY[i,,] <- segmentationArray
      for( d in seq_len( channelSize ) )
        {
        batchX[i,,,d] <- ( imageArray[,, d]   - min( imageArray[,, d] ) ) / 
                    ( max( imageArray[,, d] ) - min( imageArray[,, d] ) )
        }
      }
    # cat( "\n" )

    encodedBatchY <- encodeUnet( batchY, segmentationLabels )

    return( list( batchX, encodedBatchY ) )
    }
}
