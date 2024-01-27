import ants
import antspynet
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

tf.compat.v1.disable_eager_execution()

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

# image_size=(1024, 1024)
image_size=(224, 224)

model = antspynet.create_resnet_model_2d((None, None, 1),
   number_of_classification_labels=3,
   mode="classification",
   layers=(1, 2, 3, 4),
   residual_block_schedule=(2, 2, 2, 2), lowest_resolution=64,
   cardinality=1, squeeze_and_excite=False)

# model = tf.keras.applications.DenseNet121(include_top=False, 
#                                           weights=None, 
#                                           input_tensor=None, 
#                                           input_shape=(*image_size, 3), 
#                                           pooling='avg', 
#                                           classes=3, 
#                                           classifier_activation='softmax')

# model = antspynet.create_densenet_model_2d((*image_size, 3),
#                                           number_of_classification_labels=3,
#                                            number_of_filters=16,
#                                            depth=121,
#                                            number_of_dense_blocks=1,
#                                            growth_rate=32,
#                                            dropout_rate=0.0,
#                                           weight_decay=1e-4, 
#                                            mode='classification'
#                                            )


weights_filename = scripts_directory + "xray_flip_classification.h5"

if os.path.exists(weights_filename):
    model.load_weights(weights_filename)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])



###
#
# Set up the training generator
#

batch_size = 16 

generator = batch_generator(batch_size=batch_size,
                            image_files=train_images_list,
                            image_size=image_size)

track = model.fit(x=generator, epochs=10000, verbose=1, steps_per_epoch=100,
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


