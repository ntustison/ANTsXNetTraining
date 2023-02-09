library( keras )
library( ANTsRNet )
library( magick )
library( ANTsR )

keras::backend()$clear_session()

dataset <- 'vangogh2photo'

dataDirectory <- paste0( getwd(), "/", dataset )
if( ! dir.exists( dataDirectory ) )
  {
  zippedFile <- tempfile( pattern = dataset, fileext = ".zip" )

  if( ! file.exists( zippedFile ) )
    {
    download.file(
      url = paste0( "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/", dataset, ".zip" ),
      destfile = zippedFile )
    unzip( zippedFile )
    }
  }

inputImageSize <- c( 128, 128, 3 )

ganModel <- CycleGanModel$new(
   inputImageSize = inputImageSize )

cat( "Reading training set A files.\n" )
trainingFilesA <- list.files( paste0( dataDirectory, '/trainA/' ),
  pattern = ".jpg", full.names = TRUE )
numberOfTrainingAFiles <- length( trainingFilesA )
X_trainA <- array( data = 0, dim = c( numberOfTrainingAFiles, inputImageSize ) )
pb <- txtProgressBar( min = 0, max = length( trainingFilesA ), style = 3 )

for( i in seq_len( numberOfTrainingAFiles ) )
  {
  image <- image_read( trainingFilesA[i] )
  image <- image_data( image_scale( image, "128x128" ) )
  image <- as.double( aperm( image, c( 3, 2, 1 ) ) )
  image <- image / 127.5 - 1.0
  X_trainA[i,,,] <- as.double( image )
  setTxtProgressBar( pb, i )
  }
cat( "\nDone.\n\n" )

cat( "Reading training set B files.\n" )
trainingFilesB <- list.files( paste0( dataDirectory, '/trainB/' ),
  pattern = ".jpg", full.names = TRUE )
numberOfTrainingBFiles <- length( trainingFilesB )
X_trainB <- array( data = 0, dim = c( numberOfTrainingBFiles, inputImageSize ) )
pb <- txtProgressBar( min = 0, max = length( trainingFilesB ), style = 3 )
for( i in seq_len( numberOfTrainingBFiles ) )
  {
  image <- image_read( trainingFilesB[i] )
  image <- image_data( image_scale( image, "128x128" ) )
  image <- as.double( aperm( image, c( 3, 2, 1 ) ) )
  image <- image / 127.5 - 1.0
  X_trainB[i,,,] <- image
  setTxtProgressBar( pb, i )
  }
cat( "\nDone.\n\n" )

ganModel$train( X_trainA, X_trainB, numberOfEpochs = 200,
  batchSize = 4, sampleInterval = 1,
  sampleFilePrefix = "./CycleGanSampleImages/sample" )

