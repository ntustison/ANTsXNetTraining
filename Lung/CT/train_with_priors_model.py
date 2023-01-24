import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import glob

import random
import math

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_with_priors_generator import batch_generator

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/LungCt/'
scripts_directory = base_directory + 'Scripts/'
priors_directory = base_directory + "Data/LUNA16/PriorsResampled128X/"

image_size = (128, 128, 128)

priors = list()
for i in range(1,4):
    priors.append(ants.image_read(priors_directory + "priors" + str(i) + ".nii.gz"))

################################################
#
#  Create the model and load weights
#
################################################

classes = list(range(4))
number_of_classification_labels = len(classes)
image_modalities = ["CT", "LeftLung", "RightLung", "Airway"]
channel_size = len(image_modalities)

unet_model = antspynet.create_unet_model_3d((*image_size, channel_size),
    number_of_outputs=number_of_classification_labels, mode="classification",
    number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
    convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
    weight_decay=1e-5, additional_options=("attentionGating",))
weights_filename = scripts_directory + "lungCtWithPriorsSegmentationWeights.h5"

if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_loss = antspynet.weighted_categorical_crossentropy(weights=(1, 1, 1, 10))

# unet_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=1.0)
dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.)
surface_loss = antspynet.multilabel_surface_loss(dimensionality=3)

def combined_loss(alpha):
    def combined_loss_fixed(y_true, y_pred):
        return (alpha * dice_loss(y_true, y_pred) +
                (1 - alpha) * surface_loss(y_true, y_pred))
    return(combined_loss_fixed)

wmh_loss = combined_loss(0.5)

unet_model.compile(optimizer=keras.optimizers.Adam(),
                   loss=unet_loss,
                   metrics=['accuracy', dice_loss])

##############################################
#
# Transfer weights
#
##############################################

# unet_model_no_priors = \
#    antspynet.create_unet_model_3d((*patch_size, 1),
#    number_of_outputs=number_of_classification_labels, mode="classification",
#    number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
#    convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
#    weight_decay=1e-5,
#    additional_options=("attentionGating",))
# unet_model_no_priors.load_weights(antspynet.get_pretrained_network("sixTissueOctantBrainSegmentation")) 

# for i in range(2, len(unet_model.layers)):
#     unet_model.get_layer(index=i).set_weights(unet_model_no_priors.get_layer(index=i).get_weights())

################################################
#
#  Load the data
#
################################################

print("Loading lung data.")

ct_images = glob.glob("/home/ntustison/Data/LungCt/Data/LUNA16/NiftiResampled128X/*.nii.gz")

training_ct_files = list()
training_seg_files = list()

for i in range(len(ct_images)):
    ct = ct_images[i]
    seg = ct.replace("NiftiResampled128X", "seg-lungs-LUNA16-Resampled128X")

    if not os.path.exists(seg):
        continue

    training_ct_files.append(ct)
    training_seg_files.append(seg)

print("Total training image files: ", len(training_ct_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 8

generator=batch_generator(batch_size=batch_size,
                             image_size=image_size,
                             priors=priors,
                             images=training_ct_files,
                             segmentation_images=training_seg_files,
                             segmentation_labels=classes,
                             do_random_contralateral_flips=False,
                             do_histogram_intensity_warping=False,
                             do_add_noise=False,
                             do_data_augmentation=False
                            )

track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=48,
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



