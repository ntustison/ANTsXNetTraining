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

inputFileNames <- list.files( path = "/Users/ntustison/Pkg/UKBiobank_deep_pretrain/human_brain_image_data_preprocessed/", 
                              pattern = "example.*nii.gz", full.names = TRUE )

templateSize <- c( 160L, 192L, 160L )
fmribModel <- createSimpleFullyConvolutionalNeuralNetworkModel3D( list( NULL, NULL, NULL, 1 ) )

# brainAgeWeightsFileName <- paste0( scriptsDirectory, "/run_20190719_00_epoch_best_mae_keras.h5" )

brainAgeWeightsFileNames <- c()
brainAgeWeightsFileNames[1] <- paste0( scriptsDirectory, "/run_20190719_00_epoch_best_mae_keras_2x3x4x1x0.h5" )
brainAgeWeightsFileNames[2] <- paste0( scriptsDirectory, "/run_20190719_00_epoch_best_mae_keras_2x4x3x1x0.h5" )
brainAgeWeightsFileNames[3] <- paste0( scriptsDirectory, "/run_20190719_00_epoch_best_mae_keras_3x2x4x1x0.h5" )
brainAgeWeightsFileNames[4] <- paste0( scriptsDirectory, "/run_20190719_00_epoch_best_mae_keras_3x4x2x1x0.h5" )
brainAgeWeightsFileNames[5] <- paste0( scriptsDirectory, "/run_20190719_00_epoch_best_mae_keras_4x2x3x1x0.h5" )
brainAgeWeightsFileNames[6] <- paste0( scriptsDirectory, "/run_20190719_00_epoch_best_mae_keras_4x3x2x1x0.h5" )

brainAgeWeightsFileName <- brainAgeWeightsFileNames[1]

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
weights <- drop( fmribModel$get_layer( index = 1L )$get_weights()[[1]] )


################################################
#
#  Prediction
#
################################################

verbose <- TRUE
ageSpan <- c( 42.5, 81.5 )
numberOfBins <- 40

i <- 2
inputBrain <- antsImageRead( inputFileNames[i] )
inputBrainNormalized <- inputBrain / mean( inputBrain )

croppedBrain <- as.array( inputBrainNormalized )[11:170,13:204,11:170]
croppedBrain <- aperm( croppedBrain, c( 1, 3, 2 ) )


X <- array( data = croppedBrain, dim = c( 1, templateSize, 1 ) )
Y <- exp( drop( predict( fmribModel, X, verbose = verbose ) ) )
age <- seq( from = ageSpan[1], to = ageSpan[2], length.out = numberOfBins ) 

plotdf <- data.frame( Age = age, PredictionProbability = Y )
agePlot <- ggplot( data = plotdf ) + 
            geom_col( aes( x = Age, y = PredictionProbability ) ) + 
            ggtitle( paste0( "Age prediction: ", sum( Y * age ) ) ) +
            scale_x_continuous( breaks = seq( from = 45, to = 80, by = 5 ), labels = seq( from = 45, to = 80, by = 5 ) )
plotName <- paste0( brainAgeWeightsFileName, ".pdf" )            
ggsave( plotName, agePlot, width = 5, height = 3, units = "in" )

