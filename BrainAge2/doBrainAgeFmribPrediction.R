library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )
library( ggplot2 )

keras::backend()$clear_session()

# Sys.setenv( "CUDA_VISIBLE_DEVICES" = "1,3" )

################################################
#
#  Temporary variables:  To do
#
################################################

baseDirectory <- '/Users/ntustison/Pkg/ANTsXNetApps/BrainAgeGender/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/' )
brainAgeWeightsFileName <- paste0( scriptsDirectory, "/brainAgeFmribWeights.h5" )

################################################
#
#  Command line processing
#
################################################

# args <- commandArgs( trailingOnly = TRUE )

# if( length( args ) == 0 )
#   {
#   helpMessage <- paste0( "Usage:  Rscript doBrainAge2Prediction.R outputCsvFile inputT1_1 inputT1_2 inputT1_3 ...\n" )
#   stop( helpMessage )
#   } else {
#   outputCsvFile <- args[1]
#   inputFileNames <- args[2:length( args )]
#   }

outputCsvFile <- "None"
inputFileNames <- c( "/Users/ntustison/Pkg/UKBiobank_deep_pretrain/human_brain_image_data_preprocessed/example1_Gender0_Age45.nii.gz", 
                     "/Users/ntustison/Pkg/UKBiobank_deep_pretrain/human_brain_image_data_preprocessed/example2_Gender1_Age62.nii.gz" )

################################################
#
#  Create the model and load weights
#
################################################

templateSize <- c( 160L, 192L, 160L )
fmribModel <- createSimpleFullyConvolutionalNeuralNetworkModel3D( c( templateSize, 1 ), dropoutRate = 0.5 )

brainAgeWeightsFileName <- paste0( scriptsDirectory, "/brainAgeFmribWeights.h5" )
if( file.exists( brainAgeWeightsFileName ) )
  {
  load_model_weights_hdf5( fmribModel, filepath = brainAgeWeightsFileName )
  } else {
  stop( "Weights file doesn't exist.\n" )  
  }
fmribModel %>% compile(
  optimizer = optimizer_sgd(lr=0.01, decay=0.001, momentum=0.9, nesterov=TRUE),
  loss = "kullback_leibler_divergence",
  metrics = 'kullback_leibler_divergence' )

################################################
#
#  Prediction
#
################################################

verbose <- TRUE
ageSpan <- c( 42, 82 )
numberOfBins <- 40
deltaAge <- ( ageSpan[2] - ageSpan[1] ) / numberOfBins 

brainAges <- rep( NA, length( inputFileNames ) )
for( i in seq_len( length( inputFileNames ) ) )
  {
  inputImage <- antsImageRead( inputFileNames[i] )
  # if( verbose )
  #   {
  #   cat( "Preprocessing input image ", inputFileNames[i], ".\n", sep = '' )
  #   }
  # preprocessing <- preprocessBrainImage( inputImage, truncateIntensity = c( 0.01, 0.99 ), 
  #   doBrainExtraction = TRUE, templateTransformType = "AffineFast", template = "biobank",
  #   doBiasCorrection = TRUE, returnBiasField = FALSE, doDenoising = FALSE, 
  #   intensityMatchingType = NULL, referenceImage = NULL, 
  #   intensityNormalizationType = "01", outputDirectory = NULL, verbose = verbose )  

  # inputBrain <- preprocessing$preprocessedImage * preprocessing$brainMask 
  # inputBrainNormalized <- inputBrain %>% iMath( "Normalize" )

  inputBrainNormalized <- inputImage / mean( inputImage )

  X <- array( data = as.array( inputBrainNormalized ), dim = c( 1, templateSize, 1 ) )
  # X <- array( data = runif( prod( templateSize ) ), dim = c( 1, templateSize, 1 ) )
  Y <- drop( predict( fmribModel, X, verbose = verbose ) )
  age <- seq( from = ageSpan[1] + 0.5 * deltaAge, 
                to = ageSpan[2] - 0.5 * deltaAge, length.out = numberOfBins ) 


  brainAges[i] <- sum( age * Y )
  subjectDataFrame <- data.frame( Age = age, Prediction = Y )
  subjectPlot <- ggplot( data = subjectDataFrame ) +
                 geom_col( aes( x = Age, y = Prediction ) ) +
                 ggtitle( paste0( "Predicted age: ", brainAges[i] ) )
  ggsave( paste0( "~/Desktop/example", i, ".pdf" ), subjectPlot, width = 6, height = 4, units = "in" )                 
  }

brainAgeDataFrame <- data.frame( FileName = inputFileNames, Age = brainAgesMean )

if( outputCsvFile != "None" && outputCsvFile != "none" )
  {
  write.csv( brainAgeDataFrame, file = outputCsvFile, row.names = FALSE )
  } else {
  print( brainAgeDataFrame )
  }
