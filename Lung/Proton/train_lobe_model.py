import ants
import antspynet

from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

base_directory = '/home/ntustison/Data/Lung/Proton/'
scripts_directory = base_directory + 'Scripts/'
prior_directory = base_directory + "Data/Priors/"
template = ants.image_read(antspynet.get_antsxnet_data("protonLungTemplate"))

priors = list()
prior_labels = list(range(1, 6)) 
for p in prior_labels:
    priors.append(ants.image_read(prior_directory + "prior" + str(p) + ".nii.gz").numpy())

################################################
#
#  Create the model and load weights
#
################################################

number_of_classification_labels = 1 + len(prior_labels)
image_modalities = ["H1"]
channel_size = len(image_modalities) + len(priors)
image_size = template.shape
use_two_outputs = True

unet_model = antspynet.create_unet_model_3d((*image_size, channel_size),
   number_of_outputs=number_of_classification_labels, mode="classification", 
   number_of_filters_at_base_layer=16, number_of_layers=4,
   convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
   dropout_rate=0.0, weight_decay=0)

weights_filename = scripts_directory + "protonLungWeights.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

weighted_loss = antspynet.weighted_categorical_crossentropy(weights=(1, 10, 10, 10, 10, 10))
dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.)
binary_dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)

if use_two_outputs:
    penultimate_layer = unet_model.layers[-2].output
    outputs2 = Conv3D(filters=1,
                      kernel_size=(1, 1, 1),
                      activation='sigmoid',
                      kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)
    unet_model2 = Model(inputs=unet_model.input, outputs=[unet_model.output, outputs2])
    weights_filename = scripts_directory + "protonLungWeights2.h5"

    if os.path.exists(weights_filename):
        unet_model2.load_weights(weights_filename)

    unet_model2.compile(optimizer=keras.optimizers.Adam(lr=2e-4),
                        loss=[dice_loss, binary_dice_loss],
                        loss_weights=[0.9, 0.1])
    unet_model = unet_model2                    

else:
    def foreground_dice_loss(y_true, y_pred):
        y_true_binary = K.sum(y_true[:,:,:,:,1:], axis=4, keepdims=False)
        y_pred_binary = K.sum(y_pred[:,:,:,:,1:], axis=4, keepdims=False)
        return binary_dice_loss(y_true_binary, y_pred_binary)

    def combined_loss(alpha):
        def combined_loss_fixed(y_true, y_pred):
            return (alpha * dice_loss(y_true, y_pred) +
                    (1 - alpha) * foreground_dice_loss(y_true, y_pred))
        return(combined_loss_fixed)

    regional_and_foreground_loss = combined_loss(0.5)

    unet_model.compile(optimizer=keras.optimizers.Adam(lr=2e-4),
                    loss=regional_and_foreground_loss,
                    metrics=[foreground_dice_loss, dice_loss])

################################################
#
#  Load the data
#
################################################

print("Loading lung data.")

h1_images = glob.glob(base_directory + "Data/H1/*.nii.gz")

training_h1_files = list()
training_seg_files = list()

for i in range(len(h1_images)):
    h1_image = h1_images[i]
    seg_image = h1_image.replace("H1", "Lobes")
    training_h1_files.append(h1_image)
    training_seg_files.append(seg_image)

print("Total training image files: ", len(training_h1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 8

generator = batch_generator(batch_size=batch_size,
                            h1s=training_h1_files,
                            segmentation_images=training_seg_files,
                            priors=priors,
                            do_histogram_intensity_warping=True,
                            do_simulate_bias_field=True,
                            do_add_noise=True,
                            do_data_augmentation=True,
                            use_two_outputs=use_two_outputs
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


