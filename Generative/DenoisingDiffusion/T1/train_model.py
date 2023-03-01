import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import glob
import pandas as pd
import numpy as np
import time

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     pass

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

base_directory = '/home/ntustison/Data/Diffusion/'
template_directory = base_directory + 'Oasis/'
scripts_directory = base_directory + '/'

template = ants.image_read(antspynet.get_antsxnet_data("oasis"))

number_of_channels = 1
image_size = (64, 64, number_of_channels)

################################################
#
#  Create the U-net model, losses, and load weights (if they exist)
#
################################################

number_of_filters=(64, 128, 256, 512)
# number_of_filters=(32, 64, 128, 256)
# number_of_filters=(16, 32, 64, 128)
# number_of_filters=(8, 16, 32, 64)
# number_of_filters=(8, 16, 24, 32)

unet = antspynet.create_diffusion_probabilistic_unet_model_2d(image_size,
                                                              number_of_outputs=1,
                                                              number_of_filters=number_of_filters)
unet.compile(loss=keras.losses.MeanSquaredError(),
             optimizer=keras.optimizers.Adam(learning_rate=2e-4))
unet.summary()

weights_filename = scripts_directory + "t1_weights.h5"
ema_weights_filename = scripts_directory + "t1_ema_weights.h5"
# if os.path.exists(weights_filename):
#     unet.load_weights(weights_filename)

ema_unet = antspynet.create_diffusion_probabilistic_unet_model_2d(image_size,
                                                                  number_of_outputs=1,
                                                                  number_of_filters=number_of_filters)
ema_unet.compile(loss=keras.losses.MeanSquaredError(),
                 optimizer=keras.optimizers.Adam(learning_rate=2e-4))
ema_unet.set_weights(unet.get_weights())

optimizer=keras.optimizers.Adam(learning_rate=2e-4)

gdf_util = antspynet.GaussianDiffusion(time_steps=1000)

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = (*glob.glob("/home/ntustison/Data/CorticalThicknessData2014/*/T1/*.nii.gz"),
             *glob.glob("/home/ntustison/Data/SRPB1600/data/sub-*/t1/defaced_mprage.nii.gz"))

if len(t1_images) == 0:
    raise ValueError("NO training data.")
print("Total training image files: " + str(len(t1_images)))

print("Training")

###
#
# Set up the training generator
#

batch_size = 32

generator = batch_generator(batch_size=batch_size,
                            t1s=t1_images,
                            image_size=(image_size[0], image_size[1]),
                            number_of_channels=number_of_channels,
                            template=template,
                            do_histogram_intensity_warping=False,
                            do_simulate_bias_field=False,
                            do_add_noise=False,
                            do_data_augmentation=False
                            )

optimizer=keras.optimizers.Adam(learning_rate=1e-4)
train_acc_loss = keras.losses.MeanSquaredError()
train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()

number_of_epochs = 1000
ema=0.999

minimum_value = 1000000000

for epoch in range(number_of_epochs):
    t = tf.random.uniform(minval=0, maxval=1000, shape=(batch_size,), dtype=tf.int64)

    x_batch_train, y_batch_train = next(generator)

    with tf.GradientTape() as tape:
        # generate noise to add to the training batch
        noise = tf.random.normal(shape=x_batch_train.shape)
        # diffuse the images with noise
        x_batch_train_t = gdf_util.q_sample(x_batch_train, t, noise)
        # pass the diffused images and time steps to the network
        pred_noise = unet([x_batch_train_t, t], training=True)
        # calculate the loss
        loss_value = train_acc_loss(noise, pred_noise)

    grads = tape.gradient(loss_value, unet.trainable_weights)

    # update the weights of the network
    optimizer.apply_gradients(zip(grads, unet.trainable_weights))

    # update the weight values for the network with ema weights
    for weight, ema_weight in zip(unet.weights, ema_unet.weights):
        ema_weight.assign(ema * ema_weight + (1.0 - ema) * weight)

    # Log every 1 batches.
    if epoch % 10 == 0:
        print(
            "    Training loss (for one batch) at epoch %d: %.4f"
            % (epoch, float(loss_value))
        )
        print("        Seen so far: %d samples" % ((epoch + 1) * batch_size))
        print("Saving " + weights_filename)
        unet.save_weights(weights_filename)
        print("Saving " + ema_weights_filename)
        ema_unet.save_weights(ema_weights_filename)



