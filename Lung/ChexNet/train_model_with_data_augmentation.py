import ants
import antspynet
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#    tf.config.experimental.set_memory_growth(gpus[0], True)

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

demo2017_file = base_directory + "CXR8-selected/BBox_List_2017.csv"
demo2017 = pd.read_csv(demo2017_file)

demo_file = base_directory + "CXR8-selected/Data_Entry_2017_v2020.csv"
demo = pd.read_csv(demo_file)

def unique(list1):
    unique_list = [] 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return sorted(unique_list)

unique_labels = demo['Finding Labels'].unique()
unique_labels_unroll = []
for i in range(len(unique_labels)):
    label = unique_labels[i]
    labels = label.split('|')
    for j in range(len(labels)):
        unique_labels_unroll.append(labels[j])

unique_labels = unique(unique_labels_unroll)

training_demo_file = base_directory + "training_demo.npy"
training_demo = None
if os.path.exists(training_demo_file):
    training_demo = np.load(training_demo_file)
else:
    training_demo = np.zeros((len(train_images_list), len(unique_labels)))
    for i in tqdm(range(len(train_images_list))):
        image_filename = train_images_list[i]
        row = demo.loc[demo['Image Index'] == image_filename]
        findings = row['Finding Labels'].str.cat().split("|")
        for j in range(len(findings)):
            training_demo[i, unique_labels.index(findings[j])] = 1.0
    np.save(training_demo_file, training_demo)        

################################################
#
#  Create the model and load weights
#
################################################

number_of_classification_labels = len(unique_labels)

model = antspynet.create_resnet_model_2d((None, None, 1),
   number_of_classification_labels=number_of_classification_labels,
   mode="regression",
   layers=(1, 2, 3, 4),
   residual_block_schedule=(2, 2, 2, 2), lowest_resolution=64,
   cardinality=1, squeeze_and_excite=False)

# model = antspynet.create_resnet_model_2d((None, None, 1),
#    number_of_classification_labels=number_of_classification_labels,
#    mode="regression",
#    layers=(1, 2, 3, 4),
#    residual_block_schedule=(3, 4, 6, 3), lowest_resolution=64,
#    cardinality=1, squeeze_and_excite=False)

weights_filename = scripts_directory + "xray_classification_with_augmentation.h5"

if os.path.exists(weights_filename):
    model.load_weights(weights_filename)
else:
    flip_model = antspynet.create_resnet_model_2d((None, None, 1),
        number_of_classification_labels=3,
        mode="classification",
        layers=(1, 2, 3, 4),
        residual_block_schedule=(2, 2, 2, 2), lowest_resolution=64,
        cardinality=1, squeeze_and_excite=False)
    flip_weights_filename = scripts_directory + "xray_flip_classification.h5"
    flip_model.load_weights(flip_weights_filename)
    
    for i in range(len(model.layers)-1):
        print("Transferring layer " + str(i))
        model.get_layer(index=i).set_weights(flip_model.get_layer(index=i).get_weights())

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

###
#
# Set up the training generator
#

batch_size = 12 

generator = batch_generator(batch_size=batch_size,
                            image_files=train_images_list,
                            demo=training_demo,
                            do_augmentation=True)

track = model.fit(x=generator, epochs=10000, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95,
          verbose=1, patience=10, mode='auto')
#       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001,
#          patience=20)
       ]
   )

model.save_weights(weights_filename)


