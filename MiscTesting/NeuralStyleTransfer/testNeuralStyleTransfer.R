library( ANTsR )
library( ANTsRNet )

brain <- antsImageRead( "r16slice.nii.gz" )
starry_night_ants <- antsImageRead( "starry_night.nii.gz" )

neural <- neuralStyleTransfer( 
           brain,
           starry_night_ants,
           initialCombinationImage = NULL,
           numberOfIterations = 100,
           learningRate = 10.0,
           totalVariationWeight = 8.5e-5,
           contentWeight = 0.025,
           styleImageWeights = 1.0,
           contentLayerNames = c(
           'block5_conv2' ),
           styleLayerNames = "all",
           contentMask = NULL,
           styleMasks = NULL,
           useShiftedActivations = TRUE,
           useChainedInference = FALSE,
           verbose = TRUE,
           outputPrefix="/Users/ntustison/Desktop/NeuralTransferStyle/testR")

