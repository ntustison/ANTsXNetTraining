import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/MRIModalityClassification/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + "Nifti/"

# image_size = (224, 224, 224)
# resample_size = (1, 1, 1)

image_size = (112, 112, 112)
resample_size = (2, 2, 2)

template = ants.image_read(antspynet.get_antsxnet_data("kirby"))
template = ants.resample_image(template, resample_size)
template = antspynet.pad_or_crop_image_to_size(template, image_size)
direction = template.direction
direction[0, 0] = 1.0
ants.set_direction(template, direction)
ants.set_origin(template, (0, 0, 0))

################################################
#
#  Create the model and load weights
#
################################################

# modalities:
#     T1
#     T2
#     FLAIR
#     T2Star
#     Mean DWI
#     Mean Bold
#     ASL perfusion

number_of_classification_labels = 7
channel_size = 1

model = antspynet.create_resnet_model_3d((None, None, None, channel_size),
   number_of_classification_labels=number_of_classification_labels,
   mode="classification",
   layers=(1, 2, 3, 4),
   residual_block_schedule=(3, 4, 6, 3), lowest_resolution=64,
   cardinality=1, squeeze_and_excite=False)

weights_filename = scripts_directory + "mri_modality_classification.h5"

if os.path.exists(weights_filename):
    model.load_weights(weights_filename)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
              loss=tf.keras.losses.CategoricalCrossentropy())

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = glob.glob(data_directory + "**/*T1w.nii.gz", recursive=True)
t2_images = glob.glob(data_directory + "**/*T2w.nii.gz", recursive=True)
flair_images = glob.glob(data_directory + "**/*FLAIR.nii.gz", recursive=True)
t2star_images = glob.glob(data_directory + "**/*T2starw.nii.gz", recursive=True)
dwi_images = glob.glob(data_directory + "**/*MeanDwi.nii.gz", recursive=True)
bold_images = glob.glob(data_directory + "**/*MeanBold.nii.gz", recursive=True)
perf_images = glob.glob(data_directory + "**/*asl.nii.gz", recursive=True)

images = t1_images + t2_images + flair_images + t2star_images + dwi_images + bold_images + perf_images
modalities = np.concatenate((
             np.zeros((len(t1_images),), dtype=np.int8),
             np.zeros((len(t2_images),), dtype=np.int8) + 1,
             np.zeros((len(flair_images),), dtype=np.int8) + 2,
             np.zeros((len(t2star_images),), dtype=np.int8) + 3,
             np.zeros((len(dwi_images),), dtype=np.int8) + 4,
             np.zeros((len(bold_images),), dtype=np.int8) + 5,
             np.zeros((len(perf_images),), dtype=np.int8) + 6),
             dtype=np.int8
             )

print( "Training")

###
#
# Set up the training generator
#

batch_size = 16 

generator = batch_generator(batch_size=batch_size,
                            image_files=images,
                            template=template,
                            modalities=modalities)

track = model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
          verbose=1, patience=10, mode='auto'),
       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001,
          patience=20)
       ]
   )

model.save_weights(weights_filename)


