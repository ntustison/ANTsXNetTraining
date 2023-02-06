import ants
import antspynet

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import glob
import numpy as np
import time

from batch_generator import batch_generator

K.clear_session()

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

base_directory = '/home/ntustison/Data/IdentityUnet/'
scripts_directory = base_directory

template = ants.image_read(antspynet.get_antsxnet_data("oasis"))

################################################
#
#  Create the U-net model, losses, and load weights (if they exist)
#
################################################

channel_size = 1
image_size = (256, 256)

# unet_model = antspynet.create_unet_model_2d((*image_size, channel_size),
#                                             number_of_outputs=1,
#                                             number_of_filters=(16, 32, 64, 128),
#                                             mode="sigmoid",
#                                             dropout_rate=0.0,
#                                             convolution_kernel_size=3,
#                                             deconvolution_kernel_size=2,
#                                             weight_decay=0.0)

unet_model, input_mask = antspynet.create_partial_convolution_unet_model_2d((*image_size, channel_size),
                                                                            number_of_priors=0,
                                                                            number_of_filters=(16, 32, 64, 128),
                                                                            kernel_size=3,
                                                                            use_partial_conv=True)


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

print( "Training")

###
#
# Set up the training generator
#

batch_size = 16

generator = batch_generator(batch_size=batch_size,
                            t1s=t1_images,
                            image_size=(image_size[0], image_size[1]),
                            number_of_channels=channel_size,
                            template=template,
                            do_histogram_intensity_warping=False,
                            do_simulate_bias_field=False,
                            do_add_noise=False,
                            do_data_augmentation=False,
                            return_one_masks=True,
                            )

weights_filename = scripts_directory + "t1_pconv_identity_unet_weights.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                   loss='mse')

track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
          verbose=1, patience=10, mode='auto'),
       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001,
          patience=20)
       ]
   )
