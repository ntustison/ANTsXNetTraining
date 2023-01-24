library( ANTsR )
library( ANTsRNet )
library( keras )
library( ggplot2 )
library( tensorflow )
library( imager )


keras::backend()$clear_session()
# tf <-import("tensorflow")
# gpu_options <- tf$GPUOptions(allow_growth = TRUE)
# config <- tf$ConfigProto(gpu_options = gpu_options)
# k_set_session(tf$Session(config = config))

Sys.setenv("CUDA_VISIBLE_DEVICES" = "1")

checkImages <- FALSE

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

baseDirectory <- '/home/ntustison/Data/UNET_Aorta/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
dataDirectory <- paste0( baseDirectory, 'Data/' )

source( paste0( scriptsDirectory, 'unetBatchGenerator.R' ) )

classes <- c( "Background", "Lesion" )
imageDimensions <- c( 512, 512, 3 )
channelSize <- tail( imageDimensions, 1 )


trainingImageDirectory <- paste0( dataDirectory, 'TrainingData/OriginalImages/All_Images/' )
trainingSegmentationDirectory <- paste0( dataDirectory, 'TrainingData/Segmentations/All_Segmentations/' )

trainingSegmentationFiles <- list.files( path = trainingSegmentationDirectory,
                                         pattern = "_segment.nii.gz", full.names = TRUE )

trainingImages <- list()
trainingSegmentations <- list()
count <- 1

pb <- txtProgressBar( min = 0, max = length( trainingSegmentationFiles ), style = 3 )
for( i in seq_len( length( trainingSegmentationFiles ) ) )
{
  setTxtProgressBar( pb, i )

  segmentation <- antsImageRead( trainingSegmentationFiles[i] )
  if( max( segmentation ) < 1 )
  {
    cat( "File", trainingSegmentationFiles[i], "has no labels.\n" )
    next
  }

  trainingSegmentations[[count]] <- trainingSegmentationFiles[i]

  id <- basename( trainingSegmentationFiles[i] )
  id <- gsub( "_segment.nii.gz", '', id )

  trainingImages[[count]] <- paste0( trainingImageDirectory, id, ".tif" )
  if( ! file.exists( trainingImages[[count]] ) )
  {
    stop( paste( "File", trainingImages[[count]], "doesn't exist." ) )
  }

  if( checkImages )
  {
    segmentation <- antsImageRead( trainingSegmentations[[count]] )
    segmentationArray <- as.array( segmentation )
    imageArray <- as.array( load.image( trainingImages[[count]] ) )[,,1,, drop = TRUE]

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
      for( r in seq_len( numberOfRotations ) )
      {
        segmentationArray <- rotate90( segmentationArray )
        for( d in seq_len( channelSize ) )
        {
          imageArray[,, d] <- rotate90( imageArray[,, d] )
        }
      }
    }

    segmentation <- as.antsImage( segmentationArray )
    image <- as.antsImage( imageArray[,, 1, drop = TRUE] )
    plot( image, segmentation, alpha = 0.5 )
    Sys.sleep( 2 )
    dev.off()
  }

  count <- count + 1
}
cat( "Done.\n" )

unetModel <- createUnetModel2D( imageDimensions, numberOfOutputs = 2,
                                convolutionKernelSize = c( 3, 3 ),
                                deconvolutionKernelSize = c( 2, 2 ),
                                numberOfLayers = 4,
                                numberOfFiltersAtBaseLayer = 16,
                                weightDecay = 1e-5,
                                addAttentionGating = TRUE )

weightsFile <- paste0( scriptsDirectory, "/aortaSegmentationWeights.h5" )
if( file.exists( weightsFile ) )
{
  load_model_weights_hdf5( unetModel, weightsFile )
}

unetModel %>% compile( loss = categorical_focal_loss( alpha = 0.25, gamma = 2.0 ),
                       optimizer = optimizer_adam(),
                       metrics = c( "acc" ) )

###
#
# Set up the training generator
#

batchSize <- 48L

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfData <- length( trainingImages )
sampleIndices <- sample( numberOfData )

cat( "Total number of data = ", numberOfData, "\n" )

validationSplit <- floor( 0.8 * numberOfData )
trainingIndices <- sampleIndices[1:validationSplit]
numberOfTrainingData <- length( trainingIndices )
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfData]
numberOfValidationData <- length( validationIndices )

###
#
# Run training
#

track <- unetModel %>% fit_generator(
  generator = unetImageBatchGenerator( batchSize = batchSize,
                                       segmentationLabels = c( 0, 1 ),
                                       imageList = trainingImages[trainingIndices],
                                       segmentationList = trainingSegmentations[trainingIndices]
  ),
  steps_per_epoch = 100L,
  epochs = 200,
  validation_data = unetImageBatchGenerator( batchSize = batchSize,
                                             segmentationLabels = c( 0, 1 ),
                                             imageList = trainingImages[validationIndices],
                                             segmentationList = trainingSegmentations[validationIndices]
  ),
  validation_steps = 24,
  callbacks = list(
    callback_model_checkpoint( weightsFile,
                               monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
                               verbose = 1, mode = 'auto' ),
    callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
                                   verbose = 1, patience = 10, mode = 'auto' ),
    callback_early_stopping( monitor = 'val_loss', min_delta = 0.0001,
                             patience = 20 )
  )
)

save_model_weights_hdf5( unetModel, weightsFile )
