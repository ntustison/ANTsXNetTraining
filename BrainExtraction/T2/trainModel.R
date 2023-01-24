library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )
library( ggplot2 )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = "1" )

baseDirectory <- '/home/ntustison/Data/BrainExtractionFlairT2/'
scriptsDirectory <- paste0( baseDirectory, 'ScriptsT2/' )
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
imageModalities <- c( "T2" )
channelSize <- length( imageModalities )

unetModel <- createUnetModel3D( c( templateSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )

brainWeightsFileName <- paste0( scriptsDirectory, "/brainExtractionT2Weights.h5" )
if( file.exists( brainWeightsFileName ) )
  {
  load_model_weights_hdf5( unetModel, filepath = brainWeightsFileName )
  # } else {
  # stop( "Weights file doesn't exist.\n" )  
  }
unetModel %>% compile(
  optimizer = optimizer_adam(),
  loss = "categorical_crossentropy",
  metrics = 'accuracy' )

################################################
#
#  Load the brain data
#
################################################

cat( "Loading brain data.\n" )

brainBaseDataDirectory <- '/home/ntustison/Data/BrainExtractionFlairT2/Data/'
brainImages <- list.files( path = brainBaseDataDirectory, pattern = "*t2.nii.gz", recursive = TRUE, full.names = TRUE )

trainingImageFiles <- c()
trainingBrainMaskFiles <- c()

pb <- txtProgressBar( min = 0, max = length( brainImages ), style = 3 )
for( i in seq_len( length( brainImages ) ) )
  {
  setTxtProgressBar( pb, i )

  image <- brainImages[i]
  mask <- gsub( "-t2", "-mask", image )

  if( ! file.exists( mask ) )
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
                              doRandomContralateralFlips = FALSE,
                              doDataAugmentation = FALSE
                            ),
  steps_per_epoch = 100L,
  epochs = 200L,
  validation_data = batchGenerator( batchSize = batchSize,
                                    imageSize = templateSize, 
                                    template = template,
                                    images = trainingImageFiles[validationIndices],
                                    brainMasks = trainingBrainMaskFiles[validationIndices],
                                    doRandomContralateralFlips = FALSE,
                                    doDataAugmentation = FALSE
                                  ),  
  validation_steps = 40L,
  callbacks = list(
    callback_model_checkpoint( brainWeightsFileName,
      monitor = 'loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto' ), 
     callback_reduce_lr_on_plateau( monitor = 'loss', factor = 0.3,
       verbose = 1, patience = 10, mode = 'auto' ),
    callback_early_stopping( monitor = 'loss', min_delta = 0.001,
      patience = 20 )
  )
)

save_model_weights_hdf5( unetModel, filepath = brainWeightsFileName )
