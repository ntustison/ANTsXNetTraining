library( keras )
library( ANTsRNet )
library( tensorflow )
library( ANTsR )


tf$compat$v1$disable_eager_execution()

keras::backend()$clear_session()

# Let's use the mnist data set.

mnist <- dataset_mnist()

numberOfTrainingData <- length( mnist$train$y )

inputImageSize <- c( dim( mnist$train$x[1,,] ), 1 )


x <- array( data = ( mnist$train$x / 127.5 - 1 ),
            dim = c( numberOfTrainingData, inputImageSize ) )
y <- mnist$train$y

numberOfClusters <- length( unique( mnist$train$y ) )

# Instantiate and train the GAN model

ganModel <- ImprovedWassersteinGanModel$new(
   inputImageSize = inputImageSize,
   latentDimension = 100 )

ganModel$train( x, numberOfEpochs = 30000, batchSize = 32,
  sampleInterval = 100,
  sampleFilePrefix = "./WGanGpSampleImages/sample" )

