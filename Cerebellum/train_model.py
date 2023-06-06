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

base_directory = '/home/ntustison/Data/Cerebellum/'
scripts_directory = base_directory + 'Scripts/'
prior_directory = base_directory + "TemplateResampled0.5/"
data_directory = base_directory + "TrainingDataResampled0.5/"

priors = list()

# image_size = (233, 151, 154)
image_size = (240, 144, 144)

# Load CSF/GM/WM labels
for p in range(1, 4):
    prior = ants.image_read(prior_directory + "T_template0Posteriors" + str(p) + ".nii.gz")
    prior = antspynet.pad_or_crop_image_to_size(prior, image_size)
    priors.append(prior)

# Load regional labels
prior_labels = (*list(range(1, 14)), *list(range(100, 114)))
for p in prior_labels:
    prior = ants.image_read(prior_directory + "T_template0_label_prior_" + str(p) + ".nii.gz")
    prior = antspynet.pad_or_crop_image_to_size(prior, image_size)
    priors.append(prior)

t1_template = ants.image_read(prior_directory + "T_template0N4.nii.gz")
t1_template = antspynet.pad_or_crop_image_to_size(t1_template, image_size)

################################################
#
#  Create the model and load weights
#
################################################

number_of_classification_labels = 1 + len(prior_labels)
image_modalities = ["T1"]
channel_size = 1 + len(priors)

unet_model = antspynet.create_unet_model_3d((*image_size, channel_size),
   number_of_outputs=number_of_classification_labels, mode="classification", 
   number_of_filters=(32, 64, 96, 128, 256),
   convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
   dropout_rate=0.0, weight_decay=0,
   additional_options=None)

# weighted_loss = antspynet.weighted_categorical_crossentropy(weights=(1, 10, 10, 10, 10, 10, 10, 10))
weighted_loss_labels = antspynet.weighted_categorical_crossentropy(weights=(1, *(10,) * len(prior_labels)))
#weighted_loss_tissue = antspynet.weighted_categorical_crossentropy(weights=(10, 1, 1, 10))  #1
# weighted_loss_tissue = antspynet.weighted_categorical_crossentropy(weights=(10, 5, 1, 2))  #2 Too much csf
# weighted_loss_tissue = antspynet.weighted_categorical_crossentropy(weights=(10, 2, 1, 5))  #3
weighted_loss_tissue = antspynet.weighted_categorical_crossentropy(weights=(10, 3, 1, 5))  #4

dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.)
binary_dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)

# multiple outputs

penultimate_layer = unet_model.layers[-2].output

# whole
output2 = Conv3D(filters=1,
                 kernel_size=(1, 1, 1),
                 activation='sigmoid',
                 kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)
# csf/gm/wm
output3 = Conv3D(filters=4,
                 kernel_size=(1, 1, 1),
                 activation='softmax',
                 kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

unet_model = Model(inputs=unet_model.input, outputs=[unet_model.output, output2, output3])
weights_filename = scripts_directory + "cerebellumHierarchical.h5"

if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
                    # loss=[dice_loss, binary_dice_loss, dice_loss], 
                    loss=[weighted_loss_labels, binary_dice_loss, weighted_loss_tissue],
                    loss_weights=[0.25, 0.25, 0.5]) 
                    # loss_weights=[0.85, 0.05, 0.05])

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = glob.glob(data_directory + "*region*.nii.gz")

training_t1_files = list()
training_labels_files = list()
training_tissue_files = list()

for i in range(len(t1_images)):

    subject_directory = os.path.dirname(t1_images[i])
    labels_image = t1_images[i].replace("region", "labels")
    tissue_image = t1_images[i].replace("region", "tissue")

    training_t1_files.append(t1_images[i])
    training_labels_files.append(labels_image)
    training_tissue_files.append(tissue_image)

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 2 

generator = batch_generator(batch_size=batch_size,
                            t1s=training_t1_files,
                            t1_template=t1_template,
                            labels_images=training_labels_files,
                            tissue_images=training_tissue_files,
                            priors=priors,
                            regional_labels=prior_labels,
                            do_histogram_intensity_warping=True,
                            do_simulate_bias_field=True,
                            do_add_noise=True,
                            do_data_augmentation=True
                            )

track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
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


