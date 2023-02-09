import requests
from os import path
import ants
import antspynet
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

use_cluttered_mnist_data = False

if use_cluttered_mnist_data:

    cluttered_mnist_url = 'https://github.com/oarriaga/STN.keras/blob/master/datasets/mnist_cluttered_60x60_6distortions.npz?raw=true'
    cluttered_mnist_file = 'mnist_cluttered_60x60_6distortions.npz.npz'

    if not path.exists(cluttered_mnist_file):
        r = requests.get(cluttered_mnist_url)
        with open(cluttered_mnist_file, 'wb') as f:
            f.write(r.content)

    npz = np.load( cluttered_mnist_file )

    image_size = (60, 60)
    resampled_size = (30, 30)

    x_test = npz.f.x_test
    x_train = npz.f.x_train
    x_valid = npz.f.x_valid

    y_test = npz.f.y_test
    y_train = npz.f.y_train
    y_valid = npz.f.y_valid

else:

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = (28, 28)
    resampled_size = (64, 64)

    x_valid = x_test[0:5000,:,:]
    x_test = x_test[5000:10000,:,:]

    y_valid = keras.utils.to_categorical(y_test[0:5000])
    y_test = keras.utils.to_categorical(y_test[5000:10000])
    y_train = keras.utils.to_categorical(y_train)

##############
#
#  Set up the classification network
#

input_image_size = (*image_size, 1)
number_of_labels = 10

model = antspynet.create_simple_classification_with_spatial_transformer_network_model_2d(
  input_image_size=input_image_size, resampled_size=resampled_size,
  number_of_classification_labels = number_of_labels)

model.compile(loss='categorical_crossentropy',
  optimizer=keras.optimizers.adam(lr=0.0001),
  metrics=['categorical_crossentropy', 'accuracy'])

batch_size = 256
number_of_epochs = 300

for i in range(number_of_epochs):
    for j in range(150):
        arg0 = (j - 1) * batch_size
        arg1 = j * batch_size + 1
        x_batch = np.expand_dims(x_train[arg0:arg1,:,:], axis=-1)
        y_batch = y_train[arg0:arg1]
        loss = model.train_on_batch(x_batch, y_batch)
    validate = model.evaluate(np.expand_dims(x_valid, axis=-1), y_valid, batch_size=batch_size)
    test = model.evaluate(np.expand_dims(x_test, axis=-1), y_test, batch_size=batch_size)
    print("Epoch:", i, "| Val (acc):", validate[2], "| Test (acc):", test[2])

##############
#
#  Visualize the output of the STN
#     -> Run through a batch and plot the image before and
#        after stn-based affine transformation .
#

input_image = model.input
stn_layer = model.layers[9]
stn_function = K.function([input_image], [stn_layer.output])

for i in range(batch_size):
    print("Showing image (before)", i)
    plt.imshow(np.squeeze(x_batch[i,:,:]), cmap="Greys_r")
    plt.show()
    input("Press [enter] to continue (a)")
    out = stn_function((np.expand_dims(x_batch[i,:,:], 0)))
    print("Showing image (after)", i)
    plt.imshow(np.squeeze(out[0]), cmap="Greys_r")
    plt.show()
    input("Press [enter] to continue (b)")
