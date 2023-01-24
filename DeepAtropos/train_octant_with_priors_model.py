import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import glob

import random
import math

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_octant_with_priors_generator import batch_generator

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/SixTissueSegmentation/'
scripts_directory = base_directory + 'Scripts/'
priors_directory = base_directory + "Data/"

template_directory = '/home/ntustison/Data/BrainAge2/Data/Templates/'
template = ants.image_read(template_directory + "croppedMNI152.nii.gz")
patch_size = (112, 112, 112)

priors = list()
for i in range(7):
    priors.append(ants.image_read(priors_directory + "croppedMniPriors" + str(i) + ".nii.gz"))

################################################
#
#  Create the model and load weights
#
################################################

classes = list(range(7))
number_of_classification_labels = len(classes)
image_modalities = ["T1", "cerebellum"]
channel_size = len(image_modalities)

unet_model = antspynet.create_unet_model_3d((*patch_size, channel_size),
    number_of_outputs=number_of_classification_labels, mode="classification",
    number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
    convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
    weight_decay=1e-5, 
    additional_options=("attentionGating",))

brain_weights_filename = scripts_directory + "sixTissueOctantWithPriorsSegmentationWeights.h5"

#########
#
#          DO NOT LOAD PREVIOUS WEIGHTS!!!!!!!!!!!!!!
#
#
# if os.path.exists(brain_weights_filename):
#     unet_model.load_weights(brain_weights_filename)

unet_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 1.5, 1, 3, 4, 3, 3 ))
unet_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 1.5, 1, 3, 4, 3, 0.1 ))
unet_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 3, 1, 3, 4, 3, 0.1 ))
unet_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 10, 1, 3, 4, 3, 1 ))
unet_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 10, 1, 3, 4, 3, 0.5 )) # pretty good.  Just a tad too much WM relative to GM and DeepGM
unet_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 10, 1, 2, 4, 3, 0.5 ))

# unet_loss = antspynet.multilabel_surface_loss(dimensionality=3)

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

unet_model_no_priors = \
    antspynet.create_unet_model_3d((*patch_size, 1),
    number_of_outputs=number_of_classification_labels, mode="classification",
    number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
    convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
    weight_decay=1e-5,
    additional_options=("attentionGating",))
unet_model_no_priors.load_weights(antspynet.get_pretrained_network("sixTissueOctantBrainSegmentation")) 

for i in range(2, len(unet_model.layers)):
    unet_model.get_layer(index=i).set_weights(unet_model_no_priors.get_layer(index=i).get_weights())

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = glob.glob("/raid/data_NT/CorticalThicknessData2014/*/ThicknessAnts/*xMniInverseWarped.nii.gz")

training_t1_files = list()
training_seg_files = list()

for i in range(len(t1_images)):
    t1 = t1_images[i]
    seg = t1.replace("InverseWarped", "BrainSegmentation")

    training_t1_files.append(t1)
    training_seg_files.append(seg)

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 12

# Split trainingData into "training" and "validation" componets for
# training the model.

number_of_data = len(training_t1_files)
sample_indices = random.sample(range(number_of_data), number_of_data)

validation_split = math.floor(0.8 * number_of_data)

training_indices = sample_indices[:validation_split]
number_of_training_data = len(training_indices)

sampled_training_t1_files = list()
sampled_training_seg_files = list()

for i in range(number_of_training_data):
    sampled_training_t1_files.append(training_t1_files[training_indices[i]])
    sampled_training_seg_files.append(training_seg_files[training_indices[i]])

validation_indices = sample_indices[validation_split:]
number_of_validation_data = len(validation_indices)

sampled_validation_t1_files = list()
sampled_validation_seg_files = list()

for i in range(number_of_validation_data):
    sampled_validation_t1_files.append(training_t1_files[validation_indices[i]])
    sampled_validation_seg_files.append(training_seg_files[validation_indices[i]])


track = unet_model.fit_generator(
   generator=batch_generator(batch_size=batch_size,
                             patch_size=patch_size,
                             template=template,
                             priors=priors,
                             images=sampled_training_t1_files,
                             segmentation_images=sampled_training_seg_files,
                             segmentation_labels=classes,
                             do_random_contralateral_flips=True,
                             do_histogram_intensity_warping=False,
                             do_add_noise=False,
                             do_data_augmentation=False
                            ),
    steps_per_epoch=48,
    epochs=256,
    validation_data=batch_generator(batch_size=batch_size,
                                    patch_size=patch_size,
                                    template=template,
                                    priors=priors,
                                    images=sampled_training_t1_files,
                                    segmentation_images=sampled_training_seg_files,
                                    segmentation_labels=classes,
                                    do_random_contralateral_flips=True,
                                    do_histogram_intensity_warping=False,
                                    do_add_noise=False,
                                    do_data_augmentation=False
                                   ),
    validation_steps=24,
    callbacks=[
        keras.callbacks.ModelCheckpoint(brain_weights_filename, monitor='val_loss',
            save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
           verbose=1, patience=10, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,
           patience=20)
       ]
    )

unet_model.save_weights(brain_weights_filename)



