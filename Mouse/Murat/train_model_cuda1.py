import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR) 

from batch_generator import batch_generator

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.compat.v1.disable_eager_execution()

tf.config.run_functions_eagerly(True)


base_directory = '/home/ntustison/Data/Mouse/MuratMaga/'

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

template_file = base_directory + "image.nii.gz"
labels_file = base_directory + "labels_reduced.nii.gz"

template = ants.image_read(template_file)
labels = ants.image_read(labels_file)

new_spacing = (0.05, 0.05, 0.05)
new_shape = (256, 256, 256)

template = ants.resample_image(template, new_spacing, use_voxels=False, interp_type=4)
labels = ants.resample_image(labels, new_spacing, use_voxels=False, interp_type=1)

template = antspynet.pad_or_crop_image_to_size(template, new_shape)
labels = antspynet.pad_or_crop_image_to_size(labels, new_shape)

template = ants.iMath_normalize(template)

unique_labels = np.unique(labels.numpy())
number_of_classification_labels = len(unique_labels)
channel_size = 1

print("Unique labels: ", unique_labels)


################################################
#
#  Create the model and load weights
#
################################################

# number_of_filters = (16, 32, 64, 96, 128)
number_of_filters = (16, 32, 64, 128, 256)

unet_model = antspynet.create_unet_model_3d((*template.shape, channel_size),
   number_of_outputs=number_of_classification_labels, mode="classification", 
   number_of_filters=number_of_filters,
   convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2))

ce_loss = antspynet.weighted_categorical_crossentropy((1, *tuple([10] * (number_of_classification_labels - 1 ))))

dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.)
surface_loss = antspynet.multilabel_surface_loss()

binary_dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)
binary_surface_loss = antspynet.binary_surface_loss()

def multilabel_combined_loss(alpha=0.5):
    def multilabel_combined_loss_fixed(y_true, y_pred):
        loss = (alpha * dice_loss(y_true, y_pred) + 
                (1-alpha) * surface_loss(y_true, y_pred)) 
        return(loss)
    return multilabel_combined_loss_fixed

weights_filename = base_directory + "murat.weights.h5"
if os.path.exists(weights_filename):
    print("Loading", weights_filename)
    unet_model.load_weights(weights_filename)
    
unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
                    loss=multilabel_combined_loss(0.75),
                    metrics=[dice_loss])

###
#
# Set up the training generator
#

batch_size = 1

generator = batch_generator(batch_size=batch_size,
                    template=template,
                    images=[template],
                    labels=[labels],
                    unique_labels=unique_labels,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    resample_direction=None)

track = unet_model.fit(x=generator, epochs=100, verbose=1, steps_per_epoch=32,
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


