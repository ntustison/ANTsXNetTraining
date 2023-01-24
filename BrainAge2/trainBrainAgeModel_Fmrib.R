library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )
library( ggplot2 )


Sys.setenv( "CUDA_VISIBLE_DEVICES" = "1" )
keras::backend()$clear_session()

baseDirectory <- '/home/ntustison/Data/BrainAge2/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
source( paste0( scriptsDirectory, 'batchGenerator_Fmrib.R' ) )
templateDirectory <- paste0( baseDirectory, "Data/Templates/" )
template <- antsImageRead( paste0( templateDirectory, "croppedMNI152.nii.gz" ) )


################################################
#
#  Create the model and load weights
#
################################################

doResNet50 <- TRUE
doRegression <- TRUE
dropoutRate <- 0.0
templateSize <- c( 160L, 192L, 160L )

brainAgeWeightsFileName <- paste0( scriptsDirectory, "/brainAgeWeights_resnet50_regression.h5" )

# originalBrainAgeWeightsFileName <- '' # paste0( scriptsDirectory, "/run_20190719_00_epoch_best_mae_keras.h5" )
# if( file.exists( originalBrainAgeWeightsFileName ) )
#   {
#   load_model_weights_hdf5( brainAgeModel, filepath = originalBrainAgeWeightsFileName )
#   # } else {
#   # stop( "Weights file doesn't exist.\n" )  
#   }

if( doResNet50 )
  {
  brainAgeModel <- createResNetModel3D( c( templateSize, 1 ),
    numberOfClassificationLabels = 1, 
    layers = 1:4, residualBlockSchedule = c( 3, 4, 6, 3 ),
    lowestResolution = 64, cardinality = 1,
    mode = "regression" )
  brainAgeModel %>% compile(
    optimizer = optimizer_adam(),
    loss = tensorflow::tf$keras$losses$MeanAbsoluteError(),
    metrics = c( 'mae' ) )
  } else {
  if( doRegression == TRUE )  
    {
    brainAgeModel <- createSimpleFullyConvolutionalNeuralNetworkModel3D( list( NULL, NULL, NULL, 1 ), 
      dropout = dropoutRate, doExperimentalVariant = TRUE )
    brainAgeModel %>% compile(
      optimizer = optimizer_adam(), # optimizer_sgd( lr = 0.01, decay = 0.001, momentum = 0.9, nesterov = TRUE ),
      loss = tensorflow::tf$keras$losses$MeanAbsoluteError(),
      metrics = c( 'mae' ) )
    } else {
    brainAgeModel <- createSimpleFullyConvolutionalNeuralNetworkModel3D( list( NULL, NULL, NULL, 1 ), 
      dropout = dropoutRate, doExperimentalVariant = FALSE )
    brainAgeModel %>% compile(
      optimizer = optimizer_adam(), # optimizer_sgd( lr = 0.01, decay = 0.001, momentum = 0.9, nesterov = TRUE ),
      loss = tensorflow::tf$keras$losses$KLDivergence(),
      metrics = c( 'kullback_leibler_divergence', 'poisson', 'categorical_crossentropy' ) )
    }  
  }  

################################################
#
#  Load the brain data
#
################################################

cat( "Loading Brian data.\n" )

# brianBaseDataDirectory <- '/raid/data_BA/brains/vgg3DBrain/outputForDeepLearning/'
# brianImages <- Sys.glob( paste0( brianBaseDataDirectory, "*/*/*/*/*/*headN4Aff.nii.gz" ) )

brainBaseDataDirectory <- '/raid/data_NT/CorticalThicknessData2014/'
brainImages <- Sys.glob( paste0( brainBaseDataDirectory, "*/ThicknessAnts/*MniInverseWarped.nii.gz" ))

trainingImageFiles <- c()
trainingAges <- c()
trainingSources <- c()

pb <- txtProgressBar( min = 0, max = length( brainImages ), style = 3 )
for( i in seq_len( length( brainImages ) ) )
  {
  setTxtProgressBar( pb, i )

  image <- brainImages[i]

  csv <- ''
  source <- ''
  if( grepl( 'IXI', image ) )
    {
    csv <- sub( "xMniInverseWarped.nii.gz", "T1.csv", image )
    csv <- sub( "ThicknessAnts", "T1", csv )
    source <- 'ixi'
    } else if( grepl( 'Kirby', image ) ) {
    csv <- sub( "xMniInverseWarped.nii.gz", "MPRAGE.csv", image )
    csv <- sub( "ThicknessAnts", "T1", csv )
    source <- 'kirby'
    } else if( grepl( 'NKI', image ) ) {
    csv <- sub( "xMniInverseWarped.nii.gz", "defaced_MPRAGE.csv", image )
    csv <- sub( "ThicknessAnts", "T1", csv )
    source <- 'nki'
    } else if( grepl( 'Oasis', image ) ) {
    csv <- sub( "xMniInverseWarped.nii.gz", ".csv", image )
    csv <- sub( "ThicknessAnts", "T1", csv )
    source <- 'oasis'
    }

  if( ! file.exists( csv ) )
    {
    next  
    }

  subjectData <- read.csv( csv ) 

  age <- -1
  if( source == "ixi" || source == "kirby" )
    {
    age <- subjectData$AGE
    } else if( source == "nki" || source == "oasis" ) {
    age <- subjectData$Age
    }

  if( length( age ) != 1 || ! is.numeric( age ) || age < 42 || age > 82 )
    {
    next  
    }

  trainingImageFiles <- append( trainingImageFiles, image )
  trainingAges <- append( trainingAges, age )
  trainingSources <- append( trainingSources, source )
  }
cat( "\n" )  

trainingAges <- as.numeric( trainingAges )

cat( "**** Breakdown of data ****\n\n" )
cat( "The plots age.pdf and gender.pdf are in ", getwd(), "\n" )

trainingData <- data.frame( ImageFileName = trainingImageFiles, 
                            Age = trainingAges,
                            Source = trainingSources )

agePlot <- ggplot( trainingData ) +
  geom_density( aes( x = Age, colour = Source, fill = Source ), alpha = 0.5 )
ggsave( paste0( getwd(), "/age.pdf" ), agePlot, width = 7, height = 4, units = "in" )

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

track <- brainAgeModel %>% fit_generator(
  generator = batchGenerator( batchSize = batchSize,
                              imageSize = templateSize, 
                              numberOfBins = 40,
                              sigma = 1,
                              ageSpan = c( 42, 82 ),
                              template = template,
                              reflectionMatrix = NULL,
                              doDataAugmentation = FALSE,
                              doRegression = doRegression,
                              images = trainingImageFiles[trainingIndices],
                              ages = trainingAges[trainingIndices]
                            ),
  steps_per_epoch = 64L,
  epochs = 200L,
  validation_data = batchGenerator( batchSize = batchSize,
                                    imageSize = templateSize, 
                                    numberOfBins = 40,
                                    sigma = 1,
                                    ageSpan = c( 42, 82 ),
                                    template = template,
                                    reflectionMatrix = NULL,
                                    doDataAugmentation = FALSE,
                                    doRegression = doRegression,
                                    images = trainingImageFiles[validationIndices],
                                    ages = trainingAges[validationIndices]
                                  ),  
  validation_steps = 32L,
  callbacks = list(
    callback_model_checkpoint( brainAgeWeightsFileName,
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto' ), 
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.3,
       verbose = 1, patience = 10, mode = 'auto' ),
    callback_early_stopping( monitor = 'val_loss', min_delta = 0.001,
      patience = 20 )
  )
)

save_model_weights_hdf5( brainAgeModel, filepath = brainAgeWeightsFileName )
