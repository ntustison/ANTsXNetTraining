import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob

import random
import math

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

base_directory = '/home/ntustison/Data/Lung/'
scripts_directory = base_directory + 'Scripts/'

from batch_generator import batch_generator

template_size = (256, 256)
number_of_slices_per_image = 5
classes = tuple(list(range(5)))

################################################
#
#  Create the model and load weights
#
################################################

number_of_classification_labels = len(classes)
image_modalities = ("Vent", "Mask")
channel_size = len(image_modalities)

unet_model = antspynet.create_unet_model_2d((*template_size, channel_size),
    number_of_outputs=number_of_classification_labels, mode="classification",
    number_of_layers=4, number_of_filters_at_base_layer=32, dropout_rate=0.0,
    convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
    weight_decay=1e-5, nn_unet_activation_style=False, add_attention_gating=True)

weights_filename = scripts_directory + "ventilationWeights.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_loss = antspynet.multilabel_dice_coefficient(dimensionality=2, smoothing_factor=0.1)
# unet_loss = antspynet.multilabel_surface_loss(dimensionality=2)

unet_model.compile(optimizer=keras.optimizers.Adam(),
                   loss=unet_loss,
                   metrics=['accuracy'])


################################################
#
#  Load the data
#
################################################

print("Loading data.")

images = glob.glob("/home/ntustison/Data/Lung/Ventilation/Images/*Ventilation.nii.gz")

training_image_files = list()
training_mask_files = list()

for i in range(len(images)):

    image = images[i]
    segmentation = image.replace("Images", "Segmentations")
    segmentation = segmentation.replace("Ventilation.nii.gz", "Segmentation.nii.gz")

    if not os.path.exists(image) or not os.path.exists(segmentation):
        continue

    training_image_files.append(image)
    training_mask_files.append(segmentation)

print("Total training image files: ", len(training_image_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 128

# Split trainingData into "training" and "validation" componets for
# training the model.

number_of_data = len(training_image_files)
sample_indices = random.sample(range(number_of_data), number_of_data)

validation_split = math.floor(0.8 * number_of_data)

training_indices = sample_indices[:validation_split]
number_of_training_data = len(training_indices)

sampled_training_image_files = list()
sampled_training_mask_files = list()

for i in range(number_of_training_data):
    sampled_training_image_files.append(training_image_files[training_indices[i]])
    sampled_training_mask_files.append(training_mask_files[training_indices[i]])

validation_indices = sample_indices[validation_split:]
number_of_validation_data = len(validation_indices)

sampled_validation_image_files = list()
sampled_validation_mask_files = list()

for i in range(number_of_validation_data):
    sampled_validation_image_files.append(training_image_files[validation_indices[i]])
    sampled_validation_mask_files.append(training_mask_files[validation_indices[i]])


track = unet_model.fit_generator(
   generator=batch_generator(batch_size=batch_size,
                             image_size=template_size,
                             images=sampled_training_image_files,
                             segmentations=sampled_training_mask_files,
                             labels=classes,
                             number_of_slices_per_image=number_of_slices_per_image,
                             do_random_contralateral_flips=True,
                             do_histogram_intensity_warping=True,
                             do_add_noise=True,
                             do_data_augmentation=False
                            ),
    steps_per_epoch=64,
    epochs=256,
    validation_data=batch_generator(batch_size=batch_size,
                                    image_size=template_size,
                                    images=sampled_validation_image_files,
                                    segmentations=sampled_validation_mask_files,
                                    labels=classes,
                                    number_of_slices_per_image=number_of_slices_per_image,
                                    do_random_contralateral_flips=True,
                                    do_histogram_intensity_warping=True,
                                    do_add_noise=True,
                                    do_data_augmentation=False
                                   ),
    validation_steps=64,
    callbacks=[
        keras.callbacks.ModelCheckpoint(weights_filename, monitor='val_loss',
            save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
           verbose=1, patience=10, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,
           patience=20)
        ]
    )

unet_model.save_weights(weights_filename)



