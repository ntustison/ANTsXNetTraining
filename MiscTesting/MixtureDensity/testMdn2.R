library( keras )
library( ANTsRNet )
library( plotly )

layer_mixture_density <- function( objects,
  outputDimension, numberOfMixtures ) {
create_layer( MixtureDensityNetworkLayer, objects,
    list( outputDimension = outputDimension,
      numberOfMixtures = numberOfMixtures )
    )
}

# Generate synthetic data

numberOfSamples <- 5000

z_data <- runif( numberOfSamples, -15, 15 )
r_data <- rnorm( numberOfSamples )
s_data <- rnorm( numberOfSamples )
x_data <- sin( 0.75 * z_data ) * 7.0 + z_data * 0.5 + r_data * 1.0
y_data <- cos( 0.80 * z_data ) * 6.5 + z_data * 0.5 + s_data * 1.0

plotlyDataFrame <- data.frame( x = x_data, y = y_data, z = z_data, size = 0.2, type = "samples" )

p <- plot_ly( plotlyDataFrame, x = ~x, y = ~y, z = ~z, opacity = 0.2 ) %>%
  add_markers()

p

# buidl the MDN model

nHidden <- 15L
numberOfMixes <- 10L
outputDimension <- 2L

inputs <- layer_input( shape = c( 1 )  )

outputs <- inputs

outputs <- outputs %>% layer_dense( nHidden, activation = 'relu' )
outputs <- outputs %>% layer_dense( nHidden, activation = 'relu' )
outputs <- outputs %>% layer_mixture_density( outputDimension, numberOfMixes )

model <- keras_model( inputs = inputs, outputs = outputs )

model$compile(
  loss = getMixtureDensityLossFunction( outputDimension, numberOfMixes ),
  optimizer = optimizer_adam() )

x_input <- array( data = z_data, dim = c( numberOfSamples, 1 ) )
y_input <- as.array( cbind( x_data, y_data ) )

history <- model$fit( x = x_input, y = y_input,
  batch_size = 128L, epochs = 300L,
  validation_split=0.15, callbacks = list( callback_terminate_on_naan() ) )

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


plotlyDataFrame2 <- data.frame( x = y_samples[,1], y = y_samples[,2], z = x_test, size = 0.5, type = "predicted" )

plotlyDataFrame2 <- rbind( plotlyDataFrame, plotlyDataFrame2 )

p2 <- plot_ly( plotlyDataFrame2, x = ~x, y = ~y, z = ~z, color = ~type, opacity = 0.5 ) %>%
  add_markers()

p2


# plot the means

plotlyDataFrame3 <- plotlyDataFrame

for( i in seq_len( numberOfMixes ) )
  {
  y_samples_mu <- mus[, ( ( i - 1 ) * outputDimension ):( i * outputDimension )]

  plotlyDataFrameMu <- data.frame( x = y_samples_mu[,1], y = y_samples_mu[,2],
    z = x_test, size = 0.5, type = paste0( "predicted_mu", i ) )

  plotlyDataFrame3 <- rbind( plotlyDataFrame3, plotlyDataFrameMu )
  }

p3 <- plot_ly( plotlyDataFrame3, x = ~x, y = ~y, z = ~z, color = ~type, opacity = 0.5 ) %>%
  add_markers()

p3

# plot the variances and weights

plotlyDataFrame4 <- plotlyDataFrame

for( i in seq_len( numberOfMixes ) )
  {
  y_samples_mu <- mus[, ( ( i - 1 ) * outputDimension ):( i * outputDimension )]

  plotlyDataFrameMu <- data.frame( x = y_samples_mu[,1], y = y_samples_mu[,2],
    z = x_test, size = pis[, i], type = paste0( "predicted_mu", i ) )

  plotlyDataFrame4 <- rbind( plotlyDataFrame4, plotlyDataFrameMu )
  }

p4 <- plot_ly( plotlyDataFrame4, x = ~x, y = ~y, z = ~z, color = ~type,
  size = ~size, opacity = 0.5 ) %>%
    add_markers()

p4
