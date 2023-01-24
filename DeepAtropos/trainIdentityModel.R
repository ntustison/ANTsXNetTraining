library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )
library( ggplot2 )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = "0" )

baseDirectory <- '/home/ntustison/Data/SixTissueSegmentation/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
source( paste0( scriptsDirectory, 'batchIdentityGenerator.R' ) )

templateDirectory <- '/home/ntustison/Data/BrainAge2/Data/Templates/'
template <- antsImageRead( paste0( templateDirectory, "croppedMNI152.nii.gz" ) )
templateSize <- c( 160L, 192L, 160L )


################################################
#
#  Create the model and load weights
#
################################################

imageModalities <- c( "T1" )
channelSize <- length( imageModalities )

unetModel <- createUnetModel3D( c( templateSize, channelSize ),
   numberOfOutputs = 1, mode = 'regression',
   numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
   convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
   weightDecay = 1e-5, addAttentionGating = TRUE )

brainWeightsFileName <- paste0( scriptsDirectory, "/identity.h5" )
if( file.exists( brainWeightsFileName ) )
  {
  load_model_weights_hdf5( unetModel, filepath = brainWeightsFileName )
  # } else {
  # stop( "Weights file doesn't exist.\n" )  
  }

unetModel %>% compile(
  optimizer = optimizer_adam(),
  loss = "mse", 
  metrics = c( "mse", "mae" ) )

################################################
#
#  Load the brain data
#
################################################

cat( "Loading brain data.\n" )

brainImages <- c( 
  Sys.glob( "/raid/data_NT/CorticalThicknessData2014/*/ThicknessAnts/*xMniInverseWarped.nii.gz" ),
  Sys.glob( "/home/ntustison/Data/SixTissueSegmentation/Data/*/*xMniInverseWarped.nii.gz" )
)

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

batchSize <- 4L 

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
                              doRandomContralateralFlips = FALSE,
                              doDataAugmentation = FALSE
                            ),
  steps_per_epoch = 100L,
  epochs = 200L,
  validation_data = batchGenerator( batchSize = batchSize,
                                    imageSize = templateSize, 
                                    template = template,
                                    images = trainingImageFiles[validationIndices],
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
