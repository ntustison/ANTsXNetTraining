library( keras )
library( ANTsRNet )
library( reticulate )
library( tensorflow )

K <- keras::backend()
K$clear_session()

tf$compat$v1$disable_eager_execution()
# tf$compat$v1$reset_default_graph()
# Sys.setenv( "CUDA_VISIBLE_DEVICES" = 3 )

useClutteredMnistData <- FALSE

if( useClutteredMnistData )
  {

  clutteredMnistUrl <- 'https://github.com/oarriaga/STN.keras/blob/master/datasets/mnist_cluttered_60x60_6distortions.npz?raw=true'
  clutteredMnistFile <- 'mnist_cluttered_60x60_6distortions.npz.npz'

  if( ! file.exists( clutteredMnistFile ) )
    {
    download.file( clutteredMnistUrl, clutteredMnistFile )
    }

  np <- import( "numpy" )
  npz <- np$load( clutteredMnistFile )

  imageSize <- c( 60, 60 )
  resampledSize <- c( 30, 30 )

  Xtest <- array( data = t( npz$f[["x_test"]] ), dim = c( 10000, imageSize ) )
  Xtrain <- array( data = npz$f[["x_train"]], dim = c( 50000, imageSize ) )
  Xvalid <- array( data = npz$f[["x_valid"]], dim = c( 10000, imageSize ) )

  Ytest <- npz$f[["y_test"]]
  Ytrain <- npz$f[["y_train"]]
  Yvalid <- npz$f[["y_valid"]]

  } else {

  mnist <- dataset_mnist()

  Xtest <- mnist$test$x[1:5000,,]
  Xvalid <- mnist$test$x[5001:10000,,]
  Xtrain <- mnist$train$x

  Ytest <- to_categorical( mnist$test$y[1:5000] )
  Yvalid <- to_categorical( mnist$test$y[5001:10000] )
  Ytrain <- to_categorical( mnist$train$y )

  imageSize <- c( 28, 28 )
  resampledSize <- c( 64, 64 )
  }

##############
#
#  Set up the classification network
#

inputImageSize <- c( imageSize, 1 )
numberOfLabels <- 10

# model <- createResNetWithSpatialTransformerNetworkModel2D(
#   inputImageSize = inputImageSize,
#   resampledSize = resampledSize,
#    numberOfClassificationLabels = numberOfLabels )

model <- createSimpleClassificationWithSpatialTransformerNetworkModel2D(
  inputImageSize = inputImageSize,
  resampledSize = resampledSize,
  numberOfClassificationLabels = numberOfLabels )

model %>% compile( loss = 'categorical_crossentropy',
  optimizer = optimizer_adam( lr = 0.0001 ),
  metrics = c( 'categorical_crossentropy', 'accuracy' ) )


batchSize <- 256
numberOfEpochs <- 300

for( i in seq_len( numberOfEpochs ) )
  {
  for( j in seq_len( 150 ) )
    {
    arg0 <- ( j - 1 ) * batchSize + 1
    arg1 <- ( j ) * batchSize
    Xbatch <- array( data = Xtrain[arg0:arg1,,], dim = c( batchSize, inputImageSize ) )
    Ybatch <- Ytrain[arg0:arg1,]
    loss <- model$train_on_batch( Xbatch, Ybatch )
    }
  # if( i %% 10 == 1 )
  #   {
    valScore <- model %>% evaluate( array( data = Xvalid, dim = c( dim( Xvalid ), 1 ) ), Yvalid, verbose = 1 )
    testScore <- model %>% evaluate( array( data = Xtest, dim = c( dim( Xtest ), 1 ) ), Ytest, verbose = 1 )
    cat( "Epoch: ", i, " | Val (acc): ", valScore$acc, " | Test (acc): ", testScore$acc, "\n", sep = '' )
  #  }
  }

##############
#
#  Visualize the output of the STN
#     -> Run through a batch and plot the image before and
#        after stn-based affine transformation .
#

inputImage <- model$input
stnLayer <- model$get_layer( "layer_spatial_transformer" )
stnFunction <- K$Function( list( inputImage ), list( stnLayer$output ) )

rotate <- function( x )
  {
  t( apply( x, 2, rev ) )
  }

for( i in seq_len( batchSize ) )
  {
  cat( "Showing image (before)", i, "\n" )
  image( rotate( Xbatch[i,,,1] ) )
  readline( prompt = "Press [enter] to continue (a)" )
  out <- stnFunction( list( Xbatch[i,,,,drop = FALSE] ) )
  cat( "Showing image (after)", i, "\n" )
  image( rotate( drop( out[[1]] ) ) )
  readline( prompt = "Press [enter] to continue (b)" )
  }
