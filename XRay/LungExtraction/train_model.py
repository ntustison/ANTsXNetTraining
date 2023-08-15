import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import glob

import random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/XRayLungExtraction/'
scripts_directory = base_directory + 'Scripts/'


################################################
#
#  Create the model and load weights
#
################################################

number_of_labels = 3
image_modalities = ["XRay"]

image_size = (256, 256)
channel_size = 3

unet_model = antspynet.create_unet_model_2d((*image_size, channel_size),
   number_of_outputs=number_of_labels, mode="classification", 
   number_of_filters_at_base_layer=32, number_of_layers=4,
   convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
   dropout_rate=0.0, weight_decay=0,
   additional_options=None)

weighted_loss = antspynet.weighted_categorical_crossentropy(weights=(1, 2, 2))
dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=2, smoothing_factor=0.)

weights_filename = scripts_directory + "xrayLungExtraction.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
                    loss=dice_loss,
                    metrics=[dice_loss])

################################################
#
#  Load the data
#
################################################

print("Loading lung data.")

images = (*glob.glob(base_directory + "/**/CXR/*.nii.gz", recursive=True),
          *glob.glob(base_directory + "/**/CXR2/*.nii.gz", recursive=True),
          *glob.glob(base_directory + "/**/CXR3/*.nii.gz", recursive=True))

left_prior = ants.image_read(base_directory + "leftPrior.nii.gz", dimension=2)
right_prior = ants.image_read(base_directory + "rightPrior.nii.gz", dimension=2)

training_image_files = list()
training_label_files = list()

for i in range(len(images)):
    base_file = os.path.basename(images[i])
    local_dir = os.path.dirname(images[i])
    label_image = local_dir.replace("CXR", "Masks") + "/" + base_file
    print(images[i])
    print(label_image)
    if os.path.exists(label_image):
        training_image_files.append(images[i])
        training_label_files.append(label_image)

print("Total training image files: ", len(training_image_files))

print( "Training")


###
#
# Set up the training generator
#

batch_size = 32

generator = batch_generator(batch_size=batch_size,
                            images=training_image_files,
                            label_images=training_label_files,
                            left_prior=left_prior,
                            right_prior=right_prior,
                            do_histogram_intensity_warping=True,
                            do_simulate_bias_field=True,
                            do_add_noise=True,
                            do_data_augmentation=True
                            )

track = unet_model.fit(x=generator, epochs=2000, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95,
          verbose=1, patience=10, mode='auto'),
#       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001,
#          patience=20)
       ]
   )

unet_model.save_weights(weights_filename)


