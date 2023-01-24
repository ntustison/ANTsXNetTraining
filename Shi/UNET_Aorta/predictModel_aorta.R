#Load in libraries first time opening R
library( ANTsR )
library( ANTsRNet )
library( keras )
library( abind )
library( ggplot2 )
library(reticulate)
library( imager )

Sys.setenv("CUDA_VISIBLE_DEVICES" = "1")

for (i in 1:1)
{
  keras::backend()$clear_session()

  baseDirectory <- '/home/ntustison/Data/UNET_Aorta/'
  dataDirectory <- paste0( baseDirectory, 'Data/' )
  testingDirectory <- paste0( dataDirectory, 'TestingData/' )
  OriginalImageDirectory <- paste0( testingDirectory, 'OriginalImages/New_Testing_Images_198_Zita_349_Niki/' )
  predictedDirectory <- paste0( testingDirectory, 'PredictedSegmentations/' )

  #tell it what classes there is going to be (tell it how many segmentation images to output and of what)
  classes <- c( "Background", "Lesion" )
  segmentationLabels <- 0:( length( classes ) - 1 )

  testingImageFiles <- list.files(
    path = paste0( OriginalImageDirectory, i ),
    pattern = "*.tif", recursive = TRUE, full.names = TRUE )
  numberOfImages <- length( testingImageFiles )

  if( numberOfImages == 0 )
    {
    stop( "No images.\n" )
    }

  #Get it all set up
  testingImages <- list()
  testingMasks <- list()
  testingImageArrays <- list()
  testingMaskArrays <- list()

  imageDimensions <- c( 512, 512 )
  numberOfChannels <- 3

  X_test <- array( data = 0, dim = c( numberOfImages, imageDimensions, numberOfChannels ) )

  cat( "Reading images.\n" )

  pb <- txtProgressBar( min = 0, max = numberOfImages, style = 3 )
  referenceImage <- NULL
  for ( j in seq_len( numberOfImages ) )
  {
    setTxtProgressBar( pb, j )	  
    if( j == 1 )
      {
      referenceImage <- splitChannels( antsImageRead( testingImageFiles[j] ) )[[1]]
      }
    testimageArray <- as.array( load.image( testingImageFiles[j] ) )[,,1,, drop = TRUE]
    X_test[j,,,] <- testimageArray
  }
  cat( "\n" )

  cat( "Create model.\n" ) 
      
  unetModel <- createUnetModel2D( c( imageDimensions, numberOfChannels ), numberOfOutputs = 2,
                                  convolutionKernelSize = c( 3, 3 ),
                                  deconvolutionKernelSize = c( 2, 2 ),
                                  numberOfLayers = 4,
                                  numberOfFiltersAtBaseLayer = 16,
                                  weightDecay = 1e-5,
                                  addAttentionGating = TRUE )

  scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
  weightsFile <- paste0( scriptsDirectory, "/aortaSegmentationWeights.h5" )

  cat( "Load weights.\n" )

  load_model_weights_hdf5( unetModel,
                           filepath = weightsFile )
  
  unetModel %>% compile( loss = "categorical_crossentropy",
                             optimizer = optimizer_adam( lr = 0.0001 ),
                             metrics = c( "acc", multilabel_dice_coefficient ) )
  #Predict!
  predictedData <- unetModel %>% predict( X_test, verbose = TRUE )
  probabilityImages <- decodeUnet( predictedData, as.antsImage( X_test[1,,,1, drop = TRUE] ) )

  #Save images in predictedDirectory that was specificed above
  for( j in 1:length( probabilityImages ) )
  {
    cat( "Writing probability segmentation images for", testingImageFiles[[j]], "\n" )
    for( k in 1:length( probabilityImages[[j]] ) )
    {
      imageFileName <- gsub( ".tif",
                             paste0( "_Probability", segmentationLabels[k], ".nii.gz" ),
                             testingImageFiles[[j]] )
      imageFileName <-
        gsub( testingDirectory, predictedDirectory, imageFileName )

      dir.create( dirname( imageFileName ), showWarnings = FALSE, recursive = TRUE )

      probabilityArray <- as.array( probabilityImages[[j]][[k]] )

      antsImageWrite(
        as.antsImage( probabilityArray, reference = referenceImage ),
        imageFileName )
    }
  }
  #remove environment variables
  rm(list=ls())
}
