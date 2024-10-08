import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"

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

base_directory = '/home/ntustison/Data/Mouse/BrainExtraction/3D/'

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

template1_file = base_directory + "bspline_template.nii.gz"
mask1_file = base_directory + "bspline_template_mask.nii.gz"

template1 = ants.resample_image(ants.image_read(template1_file), (176, 176, 176), use_voxels=True, interp_type=0)
mask1 = ants.resample_image(ants.image_read(mask1_file), (176, 176, 176), use_voxels=True, interp_type=1)

ants.set_spacing(template1, (1, 1, 1))
ants.set_spacing(mask1, (1, 1, 1))

image_size = template1.shape

template1 = ants.iMath_normalize(template1)

### Template 2

template2_file = base_directory + "HR_template.nii.gz"
mask2_file = base_directory + "HR_template_mask.nii.gz"

template2 = ants.resample_image(ants.image_read(template2_file), (176, 176, 176), use_voxels=True, interp_type=0)
mask2 = ants.resample_image(ants.image_read(mask2_file), (176, 176, 176), use_voxels=True, interp_type=1)

ants.set_spacing(template2, (1, 1, 1))
ants.set_spacing(mask2, (1, 1, 1))

template2 = ants.iMath_normalize(template2)

### Template 3

template3_file = base_directory + "araikes_template.nii.gz"
mask3_file = base_directory + "araikes_template_mask.nii.gz"

template3 = ants.resample_image(ants.image_read(template3_file), (176, 176, 176), use_voxels=True, interp_type=0)
mask3 = ants.resample_image(ants.image_read(mask3_file), (176, 176, 176), use_voxels=True, interp_type=1)

ants.set_spacing(template3, (1, 1, 1))
ants.set_spacing(mask3, (1, 1, 1))

template3 = ants.iMath_normalize(template3)


################################################
#
#  Create the model and load weights
#
################################################

number_of_filters = (16, 32, 64, 128)
mode = "sigmoid"
number_of_outputs = 1

unet_model = antspynet.create_unet_model_3d((*image_size, 1),
   number_of_outputs=number_of_outputs, mode=mode, 
   number_of_filters=number_of_filters,
   convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2))

dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)

weights_filename = "mouseT2wBrainExtraction3D.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)
else:
    raise ValueError("Weights file does not exist.")

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
                    loss=dice_loss,
                    metrics=[dice_loss])

###
#
# Set up the training generator
#

batch_size = 4

generator = batch_generator(batch_size=batch_size,
                    template=template1,
                    images=[template1, template2, template3],
                    labels=[mask1, mask2, mask3],
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    resample_direction="random")

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


