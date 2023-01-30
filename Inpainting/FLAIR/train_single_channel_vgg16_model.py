import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import glob
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_vgg16_generator import batch_generator

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/Inpainting/FLAIR'
scripts_directory = base_directory

template = ants.image_read(antspynet.get_antsxnet_data("oasis"))

################################################
#
#  Create the model and load weights
#
################################################

image_modalities = ["FLAIR"]
channel_size = len(image_modalities)
template_size = template.shape

image_size=(224, 224)

vgg16_model = antspynet.create_vgg_model_2d((224, 224, channel_size),
                                             number_of_classification_labels=1,
                                             layers=(1, 2, 3, 4, 4),
                                             lowest_resolution=64,
                                             convolution_kernel_size=(3, 3),
                                             pool_size=(2, 2),
                                             strides=(2, 2),
                                             number_of_dense_units=4096,
                                             dropout_rate=0.0,
                                             style=16,
                                             mode='regression')
vgg16_model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                   loss="mse")

weights_filename = scripts_directory + "/vgg16_imagenet_single_channel_flairbrain_weights.h5"
if os.path.exists(weights_filename):
    vgg16_model.load_weights(weights_filename)
else:
    vgg16_keras = keras.applications.VGG16(include_top=True,
                                       weights='imagenet',
                                       input_tensor=None,
                                       input_shape=None,
                                       pooling=None,
                                       classes=1000,
                                       classifier_activation='softmax')
    for i in range(2, len(vgg16_keras.layers)-1):
        vgg16_model.layers[i].set_weights(vgg16_keras.layers[i].get_weights())

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

# Load open neuro data

flair_files = glob.glob("/home/ntustison/Data/OpenNeuro/Nifti/sub*/ses*/anat/*FLAIR.nii.gz")
demo = pd.read_csv("/home/ntustison/Data/OpenNeuro/Nifti/participants_ageonly.csv")

flair_images = list()
flair_ages = list()

for i in range(len(flair_files)):
    subject_id = flair_files[i].split('/')[6]
    subject_row = demo[demo['participant_id'] == subject_id]
    if subject_row.shape[0] == 1:
        flair_images.append(flair_files[i])
        flair_ages.append(int(subject_row['age']))

# Load Kirby data

flair_files = glob.glob("/home/ntustison/Data/Kirby/Images/*FLAIR*.nii.gz")
demo = pd.read_csv("/home/ntustison/Data/Kirby/kirby.csv")

for i in range(len(flair_files)):
    subject_id = os.path.basename(flair_files[i])
    subject_id = subject_id.split('-FLAIR')[0]
    subject_row = demo[demo['Subject'] == subject_id]
    if subject_row.shape[0] == 1:
        flair_images.append(flair_files[i])
        flair_ages.append(int(subject_row['Age']))


if len(flair_images) == 0:
    raise ValueError("NO training data.")
print("Total training image files: ", len(flair_images))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 32

generator = batch_generator(batch_size=batch_size,
                            t1s=flair_images,
                            t1_ages=flair_ages,
                            image_size=image_size,
                            number_of_channels=channel_size,
                            template=template,
                            do_histogram_intensity_warping=True,
                            do_simulate_bias_field=False,
                            do_add_noise=False,
                            do_data_augmentation=False
                            )


track = vgg16_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
          verbose=1, patience=10, mode='auto'),
       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001,
          patience=20)
       ]
   )

