library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )
library( ggplot2 )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = "0" )

baseDirectory <- '/home/ntustison/Data/InfantMaskData/'
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
imageModalities <- c( "T1", "T2" )
channelSize <- length( imageModalities )

unetModel <- createUnetModel3D( c( templateSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels, mode = 'classification',
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )

brainWeightsFileName <- paste0( scriptsDirectory, "/brainExtractionInfant.h5" )
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

brainBaseDataDirectory <- paste0( baseDirectory, "Nifti/" )
maskImages <- list.files( path = brainBaseDataDirectory, pattern = "mask.nii.gz", recursive = TRUE, full.names = TRUE )

trainingImageFiles <- list()
trainingBrainMaskFiles <- c()

pb <- txtProgressBar( min = 0, max = length( maskImages ), style = 3 )

count <- 1
for( i in seq_len( length( maskImages ) ) )
  {
  setTxtProgressBar( pb, i )

  mask <- maskImages[i]
  t1 <- gsub( "mask", "T1", mask )
  t2 <- gsub( "mask", "T2", mask )

  if( ! file.exists( t1 ) || ! file.exists( t2 ) || ! file.exists( mask ) )
    {
    next  
    # stop( "Mask doesn't exist." )  
    }

  # Do a quick check  
  # maskImage <- antsImageRead( mask )
  # if( sum( maskImage ) < 1000 )
  #   {
  #   cat( "mask: ", mask, "\n" )  
  #   next  
  #   }
  # imageArray <- as.array( antsImageRead( t1 ) )
  # if( any( is.na( imageArray ) ) || ( sum( imageArray ) <= 1000 ) )
  #   {
  #   cat( "image: ", t1, "\n" )  
  #   next  
  #   }
  # imageArray <- as.array( antsImageRead( t2 ) )
  # if( any( is.na( imageArray ) ) || ( sum( imageArray ) <= 1000 ) )
  #   {
  #   cat( "image: ", t2, "\n" )  
  #   next  
  #   }
    
  trainingImageFiles[[count]] <- c( t1, t2 )
  trainingBrainMaskFiles <- append( trainingBrainMaskFiles, mask )
  count <- count + 1
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
                              doDataAugmentation = TRUE
                            ),
  steps_per_epoch = 100L,
  epochs = 200L,
  validation_data = batchGenerator( batchSize = batchSize,
                                    imageSize = templateSize, 
                                    template = template,
                                    images = trainingImageFiles[validationIndices],
                                    brainMasks = trainingBrainMaskFiles[validationIndices],
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
