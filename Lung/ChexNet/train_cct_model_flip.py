import ants
import antspynet
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import glob

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator_flip import batch_generator

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/XRayCT/'
# base_directory = '/Users/ntustison/Data/Public/XRayCT/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + "Data/"

################################################
#
#  Load the data
#
################################################

train_images_file = base_directory + "CXR8-selected/train_val_list.txt"
with open(train_images_file) as f:
    train_images_list = f.readlines()
f.close()
train_images_list = [x.strip() for x in train_images_list]


################################################
#
#  Create the model and load weights
#
################################################

image_size = (32, 32)
image_size = (1024, 1024)

model = antspynet.create_vision_transformer_model_2d((*image_size, 1),
        number_of_classification_labels=2,
        number_of_attention_heads=4)
model.summary()

weights_filename = scripts_directory + "xray_flip_cct_classification.h5"

if os.path.exists(weights_filename):
    model.load_weights(weights_filename)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
              loss=tf.keras.losses.CategoricalCrossentropy())

###
#
# Set up the training generator
#

batch_size = 32 

generator = batch_generator(batch_size=batch_size,
                            image_files=train_images_list,
                            image_size=image_size)

track = model.fit(x=generator, epochs=10000, verbose=1, steps_per_epoch=64,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95,
          verbose=1, patience=20, mode='auto')
    #    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001,
    #       patience=20)
       ]
   )

model.save_weights(weights_filename)


