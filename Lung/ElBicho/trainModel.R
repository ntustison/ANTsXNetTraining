library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )
library( ggplot2 )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = "2" )

baseDirectory <- '/home/ntustison/Data/Lung/'
# baseDirectory <- '/Users/ntustison/Data/HeliumLungStudies/DeepLearning/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
source( paste0( scriptsDirectory, 'batchGenerator.R' ) )

templateSize <- c( 256L, 256L )
numberOfSlicesPerImage = 5
classes <- c( 0, 1, 2, 3, 4 )

################################################
#
#  Create the model and load weights
#
################################################

numberOfClassificationLabels <- length( classes )

imageModalities <- c( "Vent", "Mask" )
channelSize <- length( imageModalities )

unetModel <- createUnetModel2D( c( templateSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 32, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3 ), deconvolutionKernelSize = c( 2, 2 ),
  weightDecay = 1e-5, addAttentionGating = TRUE )

weightsFileName <- paste0( scriptsDirectory, "/ventilationWeights.h5" )
if( file.exists( weightsFileName ) )
  {
  load_model_weights_hdf5( unetModel, filepath = weightsFileName )
  }

weighted_loss <- weighted_categorical_crossentropy( weights = c( 1, 10, 2, 1, 1 ) )

dice_loss <- multilabel_dice_coefficient( dimensionality = 2L, smoothingFactor = 0.1 )

# metric_multilabel_dice_coefficient <-
#  custom_metric( "multilabel_dice_coefficient",
#    multilabel_dice_coefficient )

#loss_dice <- function( y_true, y_pred ) {
#   -multilabel_dice_coefficient(y_true, y_pred )
#   }
#attr(loss_dice, "py_function_name") <- "multilabel_dice_coefficient"

unetModel %>% compile(
  optimizer = optimizer_adam(),
  loss = dice_loss, # weighted_loss, categorical_focal_loss( alpha = 0.25, gamma = 2.0 ),
  metrics = c( 'accuracy', metric_categorical_crossentropy ) )

################################################
#
#  Load the data
#
################################################

cat( "Loading data.\n" )

images <- c(
  Sys.glob( "/home/ntustison/Data/Lung/Ventilation/Images/*Ventilation.nii.gz" )
)

trainingImageFiles <- c()
trainingMaskFiles <- c()

pb <- txtProgressBar( min = 0, max = length( images ), style = 3 )
for( i in seq_len( length( images ) ) )
  {
  setTxtProgressBar( pb, i )

  image <- images[i]
  segmentation <- gsub( "Images", "Segmentations", image )
  segmentation <- gsub( "Ventilation.nii.gz", "Segmentation.nii.gz", segmentation )

  if( ! file.exists( image ) )
    {
    cat( "Image: ", image )  
    stop( "Image doesn't exist." )
    next
    }

  if( ! file.exists( segmentation ) )
    {
    cat( "Mask: ", segmentation )  
    stop( "Mask doesn't exist." )
    next
    }

  trainingImageFiles <- append( trainingImageFiles, image )
  trainingMaskFiles <- append( trainingMaskFiles, segmentation )
  }
cat( "\n" )

cat( "Total training image files: ", length( trainingImageFiles ), "\n" )

cat( "\nTraining\n\n" )


###
#
# Set up the training generator
#

batchSize <- 128L

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfData <- length( trainingImageFiles )
sampleIndices <- sample( numberOfData )

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
  generator = batchGenerator( batchSize = batchSize,
                              imageSize = templateSize,
                              images = trainingImageFiles[trainingIndices],
                              segmentations = trainingMaskFiles[trainingIndices],
                              labels = classes,
                              numberOfSlicesPerImage = numberOfSlicesPerImage,
                              doRandomContralateralFlips = TRUE,
                              doHistogramIntensityWarping = TRUE,
                              doDataAugmentation = FALSE
                            ),
  steps_per_epoch = 64L,
  epochs = 256L,
  validation_data = batchGenerator( batchSize = batchSize,
                                    imageSize = templateSize,
                                    images = trainingImageFiles[validationIndices],
                                    segmentations = trainingMaskFiles[validationIndices],
                                    labels = classes,
                                    numberOfSlicesPerImage = numberOfSlicesPerImage,
                                    doRandomContralateralFlips = TRUE,
                                    doHistogramIntensityWarping = TRUE,
                                    doDataAugmentation = FALSE
                                  ),
  validation_steps = 64L,
  callbacks = list(
    callback_model_checkpoint( weightsFileName,
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto' ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
       verbose = 1, patience = 10, mode = 'auto' ),
    callback_early_stopping( monitor = 'val_loss', min_delta = 0.0000001,
      patience = 20 )
  )
)

save_model_weights_hdf5( unetModel, filepath = weightsFileName )
