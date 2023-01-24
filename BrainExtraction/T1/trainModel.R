library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )
library( ggplot2 )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = "2" )

baseDirectory <- '/home/ntustison/Data/BrainExtractionT1/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
source( paste0( scriptsDirectory, 'batchGenerator.R' ) )

templateDirectory <- paste0( baseDirectory, '../Templates/Kirby/SymmetricTemplate/' )
template <- antsImageRead( paste0( templateDirectory, "S_template3_resampled2.nii.gz" ) ) %>% iMath( "Normalize" )
templateSize <- dim( template )

################################################
#
#  Create the model and load weights
#
################################################

classes <- c( "background", "brain" )
numberOfClassificationLabels <- length( classes )
imageModalities <- c( "T1" )
channelSize <- length( imageModalities )

unetModel <- createUnetModel3D( c( templateSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )

brainWeightsFileName <- paste0( scriptsDirectory, "/brainExtraction.h5" )
if( file.exists( brainWeightsFileName ) )
  {
  load_model_weights_hdf5( unetModel, filepath = brainWeightsFileName )
  # } else {
  # stop( "Weights file doesn't exist.\n" )  
  }

# unet_loss <- 'categorical_crossentropy' # weighted_categorical_crossentropy( weights = c( 1, 1 ) )
# unet_loss <- multilabel_dice_coefficient( smoothingFactor = 1.0 ) 
unet_loss <- multilabel_dice_coefficient( smoothingFactor = 0.1 )
# unet_loss <- multilabel_dice_coefficient( smoothingFactor = 0.5 )

unetModel %>% compile(
  optimizer = optimizer_adam(),
  loss = unet_loss,
  metrics = c( metric_categorical_crossentropy, 'accuracy' ) )

################################################
#
#  Load the brain data
#
################################################

cat( "Loading brain data.\n" )

brainBaseDataDirectory <- '/home/ntustison/Data/BrainExtractionT1/Data/'
maskImages1 <- list.files( path = brainBaseDataDirectory, pattern = "*BrainExtractionMask.nii.gz", recursive = TRUE, full.names = TRUE )
maskImages2 <- list.files( path = brainBaseDataDirectory, pattern = "*ants_BrainMask.nii.gz", recursive = TRUE, full.names = TRUE )

maskImages <- c( maskImages1, maskImages2 )

trainingImageFiles <- c()
trainingBrainMaskFiles <- c()

pb <- txtProgressBar( min = 0, max = length( maskImages ), style = 3 )
for( i in seq_len( length( maskImages ) ) )
  {
  setTxtProgressBar( pb, i )

  mask <- maskImages[i]
  image <- gsub( "BrainExtractionMask", "", mask )
  image <- gsub( "_ants_BrainMask", "", mask )


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

batchSize <- 8L

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
                              brainMasks = trainingBrainMaskFiles[trainingIndices],
                              doRandomContralateralFlips = TRUE,
                              doDataAugmentation = TRUE
                            ),
  steps_per_epoch = 100L,
  epochs = 200L,
  validation_data = batchGenerator( batchSize = batchSize,
                                    imageSize = templateSize, 
                                    template = template,
                                    images = trainingImageFiles[validationIndices],
                                    brainMasks = trainingBrainMaskFiles[validationIndices],
                                    doRandomContralateralFlips = TRUE,
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
