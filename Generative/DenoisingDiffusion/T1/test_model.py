import ants
import antspynet

import glob
import os

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

K.clear_session()

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

base_directory = '/Users/ntustison/Pkg/ANTsXNetTraining/Generative/DenoisingDiffusion/T1/'

number_of_channels = 1
image_size = (256, 256, number_of_channels)

################################################
#
#  Create the U-net model, losses, and load weights (if they exist)
#
################################################

# number_of_filters=(8, 16, 32, 64)

unet = antspynet.create_diffusion_probabilistic_unet_model_2d(image_size,
                                                              number_of_filters=number_of_filters)
unet.compile(loss=keras.losses.MeanSquaredError(),
             optimizer=keras.optimizers.Adam(learning_rate=2e-4))
weights_filename = base_directory + "t1_weights.h5"
if os.path.exists(weights_filename):
    unet.load_weights(weights_filename)

ema_unet = antspynet.create_diffusion_probabilistic_unet_model_2d(image_size,
                                                                  number_of_filters=number_of_filters)
ema_unet.compile(loss=keras.losses.MeanSquaredError(),
                 optimizer=keras.optimizers.Adam(learning_rate=2e-4))
ema_weights_filename = base_directory + "t1_ema_weights.h5"
if os.path.exists(ema_weights_filename):
    ema_unet.load_weights(ema_weights_filename)
else:
    ema_unet.set_weights(unet.get_weights())

################################################
#
#  Generate the image
#
################################################

number_of_images = 16
time_steps = 1000

gdf_util = antspynet.GaussianDiffusion(time_steps=time_steps)

noise_samples = tf.random.normal(shape=(number_of_images, *image_size), dtype=tf.float32)

# 2. Sample from the model iteratively
for t in reversed(range(0, time_steps)):
    print("Time step " + str(t) + " of " + str(time_steps))
    tt = tf.cast(tf.fill(number_of_images, t), dtype=tf.int64)
    predicted_noise = ema_unet.predict([noise_samples, tt], verbose=0, batch_size=number_of_images)
    noise_samples = gdf_util.p_sample(predicted_noise, noise_samples, tt, clip_denoised=True)




