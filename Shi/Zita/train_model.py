import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import shutil

import random
import math
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/ShiData/Zita/'
scripts_directory = base_directory + 'Scripts/'

from batch_generator import batch_generator

template_size = (512, 512)
classes = (0, 1)

################################################
#
#  Create the model and load weights
#
################################################

number_of_classification_labels = len(classes)
image_modalities = ["Hist"]
channel_size = len(image_modalities)

unet_model = antspynet.create_unet_model_2d((*template_size, channel_size),
   number_of_outputs=1, mode="sigmoid", number_of_filters=(64, 96, 128, 256, 512),
   convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
   dropout_rate=0.0, weight_decay=0, 
   additional_options=("initialConvolutionKernelSize[5]", "attentionGating"))

weights_filename = scripts_directory + "weibinWeights.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

# unet_loss = antspynet.weighted_categorical_crossentropy((1, 10))

unet_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)

# unet_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)
# unet_loss = keras.losses.BinaryCrossentropy(from_logits=False)

# unet_loss = antspynet.multilabel_surface_loss(dimensionality=2)
# unet_loss = antspynet.multilabel_dice_coefficient(dimensionality=2, smoothing_factor=0.)
# surface_loss = antspynet.multilabel_surface_loss(dimensionality=2)

# def combined_loss(alpha):
#     def combined_loss_fixed(y_true, y_pred):
#         return (alpha * dice_loss(y_true, y_pred) +
#                 (1 - alpha) * surface_loss(y_true, y_pred))
#     return(combined_loss_fixed)
# wmh_loss = combined_loss(0.5)

unet_model.compile(optimizer=keras.optimizers.Adam(lr=2e-4),
                   loss=unet_loss,
                   metrics=[unet_loss])


################################################
#
#  Load the data
#
################################################

print("Loading hist data.")

hist_files = glob.glob(base_directory + "Nifti/Resampled/GoodImages_Original_Retrain/*.nii.gz")

training_hist_files = list()
training_seg_files = list()

count = 0
for i in range(len(hist_files)):
    id = (os.path.basename(hist_files[i])).replace(".tif_main.nii.gz", "")
    id = id.replace(".tif.nii.gz", "")
    seg_files = glob.glob(base_directory + "Nifti/Resampled/GoodImages_Segmented_Retrain/" + id + "*.nii.gz")

    if len(seg_files) == 0:
        continue

    training_hist_files.append(hist_files[i])
    training_seg_files.append(seg_files[0])
    count = count + 1
    if count >= 1000:
        break

print("Total training image files: ", len(training_hist_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 16 

# Split trainingData into "training" and "validation" componets for
# training the model.

number_of_data = len(training_hist_files)
number_of_training_data = int(np.round(0.8 * number_of_data))
random.seed(a=1234)
sample_indices = random.sample(range(number_of_data), number_of_training_data)

training_hist_files = np.array(training_hist_files)
training_seg_files = np.array(training_seg_files)

test_indices = list(set(range(number_of_data)) - set(sample_indices))
test_hist_files = training_hist_files[test_indices]
test_seg_files = training_seg_files[test_indices]
test_hist_directory = base_directory + "/Nifti/Testing/Images/"
test_seg_directory = base_directory + "/Nifti/Testing/Segmentations/"

# for i in range(len(test_hist_files)):
#    shutil.copy(test_hist_files[i], test_hist_directory)
#    shutil.copy(test_seg_files[i], test_seg_directory)
# raise ValueError("Done coopying test files.")


generator = batch_generator(batch_size=batch_size,
                             image_size=template_size,
                             images=training_hist_files[sample_indices],
                             segmentation_images=training_seg_files[sample_indices],
                             segmentation_labels=classes,
                             do_random_contralateral_flips=True,
                             do_histogram_intensity_warping=True,
                             do_simulate_bias_field=True,
                             do_add_noise=False
                            )

track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=64,
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

