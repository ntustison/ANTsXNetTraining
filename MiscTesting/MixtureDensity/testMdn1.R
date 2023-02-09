library( keras )
library( ANTsRNet )
library( ggplot2 )

layer_mixture_density <- function( objects,
  outputDimension, numberOfMixtures, trainable = TRUE ) {
create_layer( MixtureDensityNetworkLayer, objects,
    list( outputDimension = outputDimension,
      numberOfMixtures = numberOfMixtures, trainable = TRUE )
    )
}

# Generate synthetic data

numberOfSamples <- 3000

y_data <- runif( numberOfSamples, -10.5, 10.5 )
r_data <- rnorm( numberOfSamples )
x_data <- sin( 0.75 * y_data ) * 7.0 + y_data * 0.5 + r_data * 1.0
x_data <- array( data = x_data, dim = c( numberOfSamples, 1 ) )

plotDataFrame <- data.frame( x = x_data, y = y_data, type = "samples" )

p1 <- ggplot( data = plotDataFrame ) +
  geom_point( aes( x = x, y = y ), size = 2 )
p1


# buidl the MDN model

nHidden <- 15L
numberOfMixes <- 10L
outputDimension <- 1L
batchSize <- 128L

model <- keras_model_sequential()
model$add( layer_dense( units = nHidden, batch_input_shape = list( NULL, 1 ), activation = 'relu' ) )
model$add( layer_dense( units = nHidden, activation = 'relu' ) )
model$add( layer_mixture_density( outputDimension = outputDimension, numberOfMixtures = numberOfMixes ) )

model$compile(
  loss = getMixtureDensityLossFunction( outputDimension, numberOfMixes ),
  optimizer = optimizer_adam() )

# inputs <- layer_input( shape = c( 1 )  )

# outputs <- inputs

# outputs <- outputs %>% layer_dense( nHidden, activation = 'relu' )
# outputs <- outputs %>% layer_dense( nHidden, activation = 'relu' )
# outputs <- outputs %>% layer_mixture_density( outputDimension, numberOfMixes )

# model <- keras_model( inputs = inputs, outputs = outputs )

# model$compile(
#   loss = getMixtureDensityLossFunction( outputDimension, numberOfMixes ),
#   optimizer = optimizer_adam() )

x_input <- array( data = y_data, dim = c( numberOfSamples, 1 ) )
y_input <- x_data

history <- model$fit( x = x_input, y = y_input,
  batch_size = 128L, epochs = 500L,
  validation_split=0.15, callbacks = list( callback_terminate_on_naan() ) )

historyPlotDataFrame <- data.frame( iteration = c( history$epoch, history$epoch ),
  loss = c( unlist( history$history$loss ), unlist( history$history$val_loss ) ),
  type = c( rep( 'loss', length( history$epoch ) ), rep( 'val_loss', length( history$epoch ) ) ) )

historyPlot <- ggplot( data = historyPlotDataFrame, aes( x = iteration, y = loss, colour = type ) ) +
  geom_line()
historyPlot

# test the MDN model

x_test <- seq( from = -15.0, to = 15, by = 0.1 )
numberOfTestSamples <- length( x_test )

x_test <- array( data = x_test, dim = c( numberOfTestSamples, 1 ) )
y_test <- model %>% predict( x_test )

mus <- y_test[,1:( numberOfMixes*outputDimension )]
sigmas <- y_test[,( numberOfMixes*outputDimension + 1 ):( 2*numberOfMixes*outputDimension )]
pis <- y_test[,( 2*numberOfMixes*outputDimension + 1 ):( 2*numberOfMixes*outputDimension + numberOfMixes )]

# plot the predicted samples

y_samples <- array( data = NA, dim = c( numberOfTestSamples, outputDimension ) )

for( i in 1:nrow( y_test ) )
  {
  y_samples[i,] <- sampleFromOutput( y_test[i,, drop = FALSE], outputDimension, numberOfMixes )
  }


plotDataFrame2 <- data.frame( x = y_samples[,1], y = x_test, type = "predict" )

plotDataFrame2 <- rbind( plotDataFrame, plotDataFrame2 )

p2 <- ggplot( data = plotDataFrame2 ) +
  geom_point( aes( x = x, y = y, colour = type ), size = 2 )
p2


# plot the means

plotDataFrame3 <- plotDataFrame

for( i in seq_len( numberOfMixes ) )
  {
  y_samples_mu <- mus[, ( ( i - 1 ) * outputDimension ):( i * outputDimension ), drop = FALSE]

  plotDataFrameMu <- data.frame( x = y_samples_mu[,1], y = x_test,
    type = paste0( "predicted_mu", i-1 ) )

  plotDataFrame3 <- rbind( plotDataFrame3, plotDataFrameMu )
  }

p3 <- ggplot( data = plotDataFrame3 ) +
  geom_point( aes( x = x, y = y, colour = type ), alpha = 0.2, size = 2 )
p3
