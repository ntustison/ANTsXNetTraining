library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )
library( ggplot2 )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = "3" )

baseDirectory <- '/home/ntustison/Data/SixTissueSegmentation/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
priorsDirectory <- paste0( baseDirectory, "Data/" )
source( paste0( scriptsDirectory, 'batchOctantWithPriorsGenerator.R' ) )

templateDirectory <- '/home/ntustison/Data/BrainAge2/Data/Templates/'
template <- antsImageRead( paste0( templateDirectory, "croppedMNI152.nii.gz" ) )
patchSize <- c( 112L, 112L, 112L )

priorFiles <- list.files( path = priorsDirectory, pattern = "croppedMniPriors",
			  recursive = FALSE, full.names = TRUE )

cat( "Readin gpriors\n" )
priors <- list()
for( i in seq.int( length( priorFiles ) ) )
  {
  priors[[i]] <- antsImageRead( priorFiles[i] )
  }


################################################
#
#  Create the model and load weights
#
################################################

cat( "Create u-net\n" )

classes <- c( 0:6 ) 
numberOfClassificationLabels <- length( classes )
imageModalities <- c( "T1", "background", "csf", "gm", "wm", "deepGm", "brainStem", "cerebellum" )
channelSize <- length( imageModalities )

unetModel <- createUnetModel3D( c( patchSize, channelSize ),
   numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
   numberOfLayers = 4, numberOfFiltersAtBaseLayer = 16, dropoutRate = 0.0,
   convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
   weightDecay = 1e-5, additionalOptions = c( "attentionGating" ) )

cat( "HERE0\n" )


brainWeightsFileName <- paste0( scriptsDirectory, "/sixTissueOctantWithPriorsSegmentationWeights.h5" )
if( file.exists( brainWeightsFileName ) )
  {
  load_model_weights_hdf5( unetModel, filepath = brainWeightsFileName )
  # } else {
  # stop( "Weights file doesn't exist.\n" )  
  }

# weighted_loss <- weighted_categorical_crossentropy( weights = c( 0.05, 1, 2, 2, 3, 2, 2 ) )
# weighted_loss <- weighted_categorical_crossentropy( weights = c( 0.05, 1, 1, 2, 3, 2, 2 ) )
# weighted_loss <- weighted_categorical_crossentropy( weights = c( 0.05, 2, 1, 3, 4, 3, 3 ) )
# way too much csf --- weighted_loss <- weighted_categorical_crossentropy( weights = c( 0.05, 4, 1, 5, 5, 3, 3 ) )
# slightly too much csf weighted_loss <- weighted_categorical_crossentropy( weights = c( 0.05, 2.5, 1, 4, 4, 3, 3 ) )
# weighted_loss <- weighted_categorical_crossentropy( weights = c( 0.05, 2, 1, 5, 5, 3, 3 ) )
weighted_loss <- weighted_categorical_crossentropy( weights = c( 0.05, 1.5, 1, 3, 4, 3, 3 ) )

# weighted_loss <- weighted_categorical_crossentropy( weights = c( 0.0025, 0.045, 0.021, 0.026, 0.26, 0.56, 0.08 ) )

metric_multilabel_dice_coefficient <-
  custom_metric( "multilabel_dice_coefficient",
    multilabel_dice_coefficient )

loss_dice <- function( y_true, y_pred ) {
   -multilabel_dice_coefficient(y_true, y_pred)
   }
attr(loss_dice, "py_function_name") <- "multilabel_dice_coefficient"

unetModel %>% compile(
  optimizer = optimizer_adam(),
  loss = weighted_loss # tensorflow::tf$keras$losses$CategoricalCrossentropy(),
) #  metrics = c( metric_multilabel_dice_coefficient, metric_categorical_crossentropy ) )


################################################
#
#  Load the brain data
#
################################################

cat( "Loading brain data.\n" )

brainImages <-  
  Sys.glob( "/raid/data_NT/CorticalThicknessData2014/*/ThicknessAnts/*xMniInverseWarped.nii.gz" )


trainingImageFiles <- c()
trainingBrainMaskFiles <- c()

pb <- txtProgressBar( min = 0, max = length( brainImages ), style = 3 )
for( i in seq_len( length( brainImages ) ) )
  {
  setTxtProgressBar( pb, i )

  image <- brainImages[i]
  mask <- gsub( "InverseWarped", "BrainSegmentation", image )

  if( ! file.exists( image ) || ! file.exists( mask ) )
    {
    next  
    # stop( "Mask doesn't exist." )  
    }

  # # Do a quick check  
  # maskImage <- antsImageRead( mask )
  # if( sum( maskImage ) < 1000 )
  #   {
  #   cat( "mask: ", mask, "\n" )  
  #   next  
  #   }
  # imageArray <- as.array( antsImageRead( image ) )
  # if( any( is.na( imageArray ) ) || ( sum( imageArray ) <= 1000 ) )
  #   {
  #   cat( "image: ", image, "\n" )  
  #   next  
  #   }
    
  trainingImageFiles <- append( trainingImageFiles, image )
  trainingBrainMaskFiles <- append( trainingBrainMaskFiles, mask )
  }
cat( "\n" )  

cat( "Total training image files: ", length( trainingImageFiles ), "\n" )

cat( "\nTraining\n\n" )

###
#
# Set up the training generator
#

batchSize <- 12L 

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
                              patchSize = patchSize, 
                              template = template,
			      priors = priors,
                              images = trainingImageFiles[trainingIndices],
                              segmentationImages = trainingBrainMaskFiles[trainingIndices],
			      segmentationLabels = classes,
                              doRandomContralateralFlips = FALSE,
                              doDataAugmentation = FALSE
                            ),
  steps_per_epoch = 100L,
  epochs = 200L,
  validation_data = batchGenerator( batchSize = batchSize,
                                    patchSize = patchSize, 
                                    template = template,
				    priors = priors,
                                    images = trainingImageFiles[validationIndices],
                                    segmentationImages = trainingBrainMaskFiles[validationIndices],
				    segmentationLabels = classes,
                                    doRandomContralateralFlips = FALSE,
                                    doDataAugmentation = FALSE
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
