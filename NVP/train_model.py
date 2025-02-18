import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"

import glob

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator
from create_normalizing_flow_model import create_normalizing_flow_model

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)

# tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/NVP/'
scripts_directory = base_directory + 'Scripts/'

data_directory = '/home/ntustison/Data/Inpainting/AlignedToNKISlices/'
brain_slice_files = glob.glob(data_directory + "*/Images/axial/*.nii.gz")

print("Total number of training brain slices: ", str(len(brain_slice_files)))

################################################
#
#  Create the model and load weights
#
################################################

# Copy from https://github.com/aganse/flow_models/blob/main/train.py

image_size = (256, 256, 1)

nvp_model = create_normalizing_flow_model((image_size), 
    hidden_layers=[512, 512], flow_steps=6, regularization=0.0,
    validate_args=False)
nvp_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
                  loss='negative_log_likelihood')

weights_filename = scripts_directory + "nvp_t1_axial.h5"

if os.path.exists(weights_filename):
    print("Loading " + weights_filename)
    nvp_model.load_weights(weights_filename)

###
#
# Set up the training generator
#

batch_size = 64

generator = batch_generator(batch_size=batch_size,
                            input_image_files=brain_slice_files,
                            image_size=image_size,
                            do_histogram_intensity_warping=True,
                            do_simulate_bias_field=False,
                            do_add_noise=False,
                            do_random_transformation=False,
                            do_random_contralateral_flips=False,
                            do_resampling=False,
                            verbose=False)

track = nvp_model.fit(x=generator, epochs=10, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='negative_log_likelihood',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='negative_log_likelihood', factor=0.95,
          verbose=1, patience=20, mode='auto'),
    #    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001,
    #       patience=20)
       ]
   )

nvp_model.save_weights(weights_filename)


