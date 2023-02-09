import numpy as np
import keras
import keras.backend as K

from keras.datasets import mnist

import ants
import antspynet

K.clear_session()

# Let's use the mnist data set.

(x_train, y_train), (x_test, y_test) = mnist.load_data()

number_of_training_data = len(y_train)
input_image_size = (28, 28, 1)

x = x_train / 127.5 - 1.0
x = np.expand_dims(x, axis=-1)
y = y_train

number_of_clusters = len( np.unique( y ) )

# Instantiate and train the GAN model

gan_model = antspynet.WassersteinGanModel(
   input_image_size=input_image_size, latent_dimension=100)

gan_model.train(x, number_of_epochs=30000, sample_interval=100,
  sample_file_prefix="./WassersteinGanSampleImages_Py/sample" )

