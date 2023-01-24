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

base_directory = '/home/ntustison/Data/BrainExtractionT1/'
scripts_directory = base_directory + 'Scripts/'

from batch_generator import batch_generator

template = ants.image_read(base_directory + 'S_template3.nii.gz')
template_brain_mask = ants.image_read(base_directory + 'S_templateBrainMask.nii.gz')
template_size = template.shape

################################################
#
#  Create the model and load weights
#
################################################

classes = ['background', 'brain']
number_of_classification_labels = len(classes)
image_modalities = ["T1", "COM_DIST"]
channel_size = len(image_modalities)

unet_model = antspynet.create_unet_model_3d((*template_size, channel_size),
    number_of_outputs=number_of_classification_labels,
    number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
    convolution_kernel_size=5, deconvolution_kernel_size=3,
    weight_decay=1e-5)

weights_filename = scripts_directory + "brainExtraction.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_loss = antspynet.weighted_categorical_crossentropy(weights=(1, 1))
# unet_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.5)
# unet_loss = antspynet.multilabel_surface_loss(dimensionality=3)

dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.)


unet_model.compile(optimizer=keras.optimizers.Adam(),
                   loss=unet_loss,
                   metrics=['accuracy', dice_loss])


################################################
#
#  Load the brain data
#
################################################

print("Loading braindata.")

base_data_directory = base_directory + '/Data/'
mask_images_1 = glob.glob(base_data_directory + "CorticalThicknessData2014/*/*HeadMask.nii.gz")
mask_images_2 = glob.glob(base_data_directory + "Oasis3BrainExtractionProcessed/*/*/*/*ants_HeadMask.nii.gz")
mask_images = (*mask_images_1, *mask_images_2)
# mask_images = mask_images_1

training_image_files = list()
training_mask_files = list()

for i in range(len(mask_images)):
    mask = mask_images[i]
    image = mask.replace("_ants_HeadMask", "")
    image = image.replace("HeadMask", "")

    if not os.path.exists(image) or not os.path.exists(mask):
        # print(mask + " ---> " + image)
        continue

    training_image_files.append(image)
    training_mask_files.append(mask)

print("Total training image files: ", len(training_image_files))
print( "Training")


###
#
# Set up the training generator
#

batch_size = 8

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
                             template=template,
                             template_brain_mask=template_brain_mask,
                             image_size=template_size,
                             images=sampled_training_image_files,
                             brain_masks=sampled_training_mask_files,
                             do_random_contralateral_flips=False,
                             do_histogram_intensity_warping=False,
                             do_add_noise=False,
                             do_data_augmentation=False
                            ),
    steps_per_epoch=64,
    epochs=256,
    validation_data=batch_generator(batch_size=batch_size,
                                    template=template,
                                    template_brain_mask=template_brain_mask,
                                    image_size=template_size,
                                    images=sampled_validation_image_files,
                                    brain_masks=sampled_validation_mask_files,
                                    do_random_contralateral_flips=False,
                                    do_histogram_intensity_warping=False,
                                    do_add_noise=False,
                                    do_data_augmentation=False
                                   ),
    validation_steps=32,
    callbacks=[
        keras.callbacks.ModelCheckpoint(weights_filename, monitor='val_loss',
            save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
           verbose=1, patience=10, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
           patience=20)
        ]
    )

unet_model.save_weights(weights_filename)



