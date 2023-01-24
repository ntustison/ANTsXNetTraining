library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )
library( ggplot2 )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = "1" )

baseDirectory <- '/home/ntustison/Data/DktMalfLabeling/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
source( paste0( scriptsDirectory, 'batchGenerator.R' ) )

templateDirectory <- '/home/ntustison/Data/BrainAge2/Data/Templates/'
template <- antsImageRead( paste0( templateDirectory, "croppedMNI152.nii.gz" ) )
templateSize <- c( 160L, 192L, 160L )

################################################
#
#  Create the model and load weights
#
################################################

#classes <- c( 0, 4, 6, 7, 10:18, 24, 26, 28, 30, 43, 44:46, 49:54, 58, 60, 91:92, 
#              630:632, 1002:1003, 1005:1031, 1034:1035, 2002:2003, 2005:2031, 2034:2035 )

classes <- c( 0, 4, 6, 7, 10:18, 24, 26, 28, 30, 43, 44:46, 49:54, 58, 60, 91:92,
              630:632 )

numberOfClassificationLabels <- length( classes )
imageModalities <- c( "T1" )
channelSize <- length( imageModalities )

unetModel <- createUnetModel3D( c( templateSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 0.0, addAttentionGating = TRUE )

brainWeightsFileName <- paste0( scriptsDirectory, "/dktLabeling.h5" )
if( file.exists( brainWeightsFileName ) )
  {
  load_model_weights_hdf5( unetModel, filepath = brainWeightsFileName )
  # } else {
  # stop( "Weights file doesn't exist.\n" )  
  }

metric_multilabel_dice_coefficient <-
  custom_metric( "multilabel_dice_coefficient",
    multilabel_dice_coefficient )

loss_dice <- function( y_true, y_pred ) {
   -multilabel_dice_coefficient(y_true, y_pred)
   }
attr(loss_dice, "py_function_name") <- "multilabel_dice_coefficient"

unetModel %>% compile(
  optimizer = optimizer_adam(),
  loss = loss_dice, # weighted_loss, # tensorflow::tf$keras$losses$CategoricalCrossentropy(),
  metrics = c( metric_multilabel_dice_coefficient, metric_categorical_crossentropy ) )

# unetModel %>% compile(
#  optimizer = optimizer_adam(),
#  loss = categorical_focal_loss( alpha = 0.25, gamma = 2.0 ),
#  metrics = 'accuracy' )

################################################
#
#  Load the brain data
#
################################################

cat( "Loading brain data.\n" )

brainImages <- c( 
  Sys.glob( "/raid/data_NT/CorticalThicknessData2014/*/ThicknessAnts/*xMniInverseWarped.nii.gz" ),
  Sys.glob( "/home/ntustison/Data/SixTissueSegmentation/Data/*/*xMniInverseWarped.nii.gz" ) )

trainingImageFiles <- c()
trainingSegmentationImageFiles <- c()

pb <- txtProgressBar( min = 0, max = length( brainImages ), style = 3 )
for( i in seq_len( length( brainImages ) ) )
  {
  setTxtProgressBar( pb, i )

  image <- brainImages[i]
  seg <- gsub( "InverseWarped", "RefinedMalfLabels", image )

  if( ! file.exists( image ) || ! file.exists( seg ) )
    {
    next  
    # stop( "Mask doesn't exist." )  
    }

  trainingImageFiles <- append( trainingImageFiles, image )
  trainingSegmentationImageFiles <- append( trainingSegmentationImageFiles, seg )
  }
cat( "\n" )  

cat( "Total training image files: ", length( trainingImageFiles ), "\n" )

cat( "\nTraining\n\n" )

###
#
# Set up the training generator
#

batchSize <- 1L

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
                              template = template,
                              images = trainingImageFiles[trainingIndices],
                              segmentationImages = trainingSegmentationImageFiles[trainingIndices],
                              segmentationLabels = classes,
                              doRandomContralateralFlips = FALSE,
                              doDataAugmentation = TRUE 
                            ),
  steps_per_epoch = 100L,
  epochs = 200L,
  validation_data = batchGenerator( batchSize = batchSize,
                                    imageSize = templateSize, 
                                    template = template,
                                    images = trainingImageFiles[validationIndices],
                                    segmentationImages = trainingSegmentationImageFiles[validationIndices],
                                    segmentationLabels = classes,
                                    doRandomContralateralFlips = FALSE,
                                    doDataAugmentation = TRUE
                                  ),  
  validation_steps = 40L,
  callbacks = list(
    callback_model_checkpoint( brainWeightsFileName,
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto' ), 
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.3,
       verbose = 1, patience = 10, mode = 'auto' ),
    callback_early_stopping( monitor = 'val_loss', min_delta = 0.001,
      patience = 20 )
  )
)

save_model_weights_hdf5( unetModel, filepath = brainWeightsFileName )
