import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"
os.environ['TF_USE_LEGACY_KERAS'] = "True"

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.compat.v1.disable_eager_execution()

# tf.config.run_functions_eagerly(True)

base_directory = '/home/ntustison/Data/Ferret/BrainExtraction/'

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

template_file = base_directory + "S_template0.nii.gz"
mask_file = base_directory + "S_template_mask.nii.gz"

template = ants.image_read(template_file)
mask = ants.image_read(mask_file)

template = ants.iMath_normalize(template)


################################################
#
#  Create the model and load weights
#
################################################

number_of_filters = (16, 32, 64, 128)
mode = "classification"
number_of_outputs = 2
number_of_classification_labels = 2

unet_model = antspynet.create_unet_model_3d((*template.shape, 1),
   number_of_outputs=number_of_outputs, mode=mode, 
   number_of_filters=number_of_filters,
   convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2))

dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)
ce_loss = tf.keras.losses.BinaryCrossentropy()


dice_loss = antspynet.multilabel_dice_coefficient(dimensionality = 3, smoothing_factor=0.0)
ce_loss = antspynet.weighted_categorical_crossentropy((1, *tuple([10] * (number_of_classification_labels - 1 ))))



weights_filename = "ferretT1wBrainExtraction3D.weights.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
                    loss=dice_loss,
                    metrics=[dice_loss])

###
#
# Set up the training generator
#

batch_size = 4

generator = batch_generator(batch_size=batch_size,
                    template=template,
                    images=[template],
                    labels=[mask],
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    resample_direction=None)

track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95,
          verbose=1, patience=20, mode='auto'),
    #    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001,
    #       patience=20)
       ]
   )

unet_model.save_weights(weights_filename)


