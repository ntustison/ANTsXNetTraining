import ants
import antspynet

from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob

import random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator_both import batch_generator

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/DeepFlash/'
scripts_directory = base_directory + 'Scripts/'
prior_directory = base_directory + "Data/Priors/"

priors = list()
prior_labels = list(range(6, 19, 2))
for p in prior_labels:
    priors.append(ants.image_read(prior_directory + "prior" + str(p) + ".nii.gz"))

t1_template = ants.image_read(base_directory + "Data/deepFlashTemplateT1SkullStripped.nii.gz")
t2_template = ants.image_read(base_directory + "Data/deepFlashTemplateT2SkullStripped.nii.gz")

################################################
#
#  Create the model and load weights
#
################################################

number_of_classification_labels = 1 + len(prior_labels)
image_modalities = ["T1", "T2"]
channel_size = len(image_modalities) + len(priors)
image_size = (64, 64, 96)
use_two_outputs = True

unet_model = antspynet.create_unet_model_3d((*image_size, channel_size),
   number_of_outputs=number_of_classification_labels, mode="classification", 
   number_of_filters=(32, 64, 96, 128, 256),
   convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
   dropout_rate=0.0, weight_decay=0,
   additional_options=None)

weighted_loss = antspynet.weighted_categorical_crossentropy(weights=(1, 10, 10, 10, 10, 10, 10, 10))
dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.)
binary_dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)

# multiple outputs

penultimate_layer = unet_model.layers[-2].output

# whole
output2 = Conv3D(filters=1,
                 kernel_size=(1, 1, 1),
                 activation='sigmoid',
                 kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)
#hippocampus
output3 = Conv3D(filters=1,
                 kernel_size=(1, 1, 1),
                 activation='sigmoid',
                 kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

# entorhinal, etc.                  
output4 = Conv3D(filters=1,
                 kernel_size=(1, 1, 1),
                 activation='sigmoid',
                 kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

unet_model = Model(inputs=unet_model.input, outputs=[unet_model.output, output2, output3, output4])
weights_filename = scripts_directory + "deepFlashRightBothHierarchical.h5"

if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_model.compile(optimizer=keras.optimizers.Adam(lr=2e-4),
                    loss=[dice_loss, binary_dice_loss, binary_dice_loss, binary_dice_loss],
                    loss_weights=[0.85, 0.05, 0.05, 0.05])
                    # round1: loss_weights=[0.5, 0.2, 0.15, 0.15]) 

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

# t1_images = (*glob.glob(base_directory + "Data/Nifti*/S*/deepFlashTemplatexstruct.nii.gz"),
#              *glob.glob(base_directory + "Data/Nifti*/S*/Templates/*/deepFlashTemplatexstruct.nii.gz")  )

t1_images = (*glob.glob(base_directory + "Data/Nifti*/S*/deepFlashTemplatexstruct.nii.gz"),
             *glob.glob(base_directory + "Data/Nifti*/S*/Templates/MCI/deepFlashTemplatexstruct.nii.gz"),
             *glob.glob(base_directory + "Data/Nifti*/S*/Templates/ADNI/deepFlashTemplatexstruct.nii.gz"),
             *glob.glob(base_directory + "Data/Nifti*/S*/Templates/*MALE*80_90/deepFlashTemplatexstruct.nii.gz")
            )

training_t1_files = list()
training_t2_files = list()
training_seg_files = list()

for i in range(len(t1_images)):

    subject_directory = os.path.dirname(t1_images[i])
    t2_image = t1_images[i].replace("struct", "T2_struct")
    seg_image = t1_images[i].replace("struct", "sROIsubEC")

    training_t1_files.append(t1_images[i])
    training_t2_files.append(t2_image)
    training_seg_files.append(seg_image)

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 8

generator = batch_generator(batch_size=batch_size,
                            t1s=training_t1_files,
                            t1_template=t1_template,
                            t2s=training_t2_files,
                            t2_template=t2_template,
                            segmentation_images=training_seg_files,
                            which_hemi="right",
                            priors=priors,
                            do_histogram_intensity_warping=True,
                            do_simulate_bias_field=True,
                            do_add_noise=True,
                            do_data_augmentation=True
                            )

track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=50,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
          verbose=1, patience=10, mode='auto'),
       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001,
          patience=20)
       ]
   )

unet_model.save_weights(weights_filename)


