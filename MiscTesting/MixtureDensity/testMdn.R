library( keras )

layer_mixture_density <- function( objects,
  outputDimension, numberOfMixtures ) {
create_layer( MixtureDensityNetworkLayer, objects,
    list( outputDimension = outputDimension,
      numberOfMixtures = numberOfMixtures )
    )
}

nHidden <- 5
nMixes <- 5

inputs <- layer_input( shape = c( 10 )  )

outputs <- inputs

outputs <- outputs %>% layer_dense( nHidden, activation = 'relu' )
outputs <- outputs %>% layer_dense( nHidden, activation = 'relu' )
outputs <- outputs %>% layer_mixture_density( 1, nMixes )

model <- keras_model( inputs = inputs, outputs = outputs )

model$compile( loss = getMixtureDensityLossFunction( 1, nMixes ), optimizer = optimizer_adam() )

