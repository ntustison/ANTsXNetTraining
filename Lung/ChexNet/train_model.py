import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/reproduce-chexnet/'
scripts_directory = base_directory + 'antsxnet/scripts/'
data_directory = base_directory + "data/"

image_size = (224, 224)

################################################
#
#  Load the data
#
################################################

demo = pd.read_csv(base_directory + "nih_labels.csv", index_col=0)

################################################
#
#  Create the model and load weights
#
################################################

number_of_classification_labels = 14
number_of_channels = 3

model = tf.keras.applications.DenseNet121(include_top=False, 
                                          weights="imagenet", 
                                          input_tensor=None, 
                                          input_shape=(224, 224, 3), 
                                          pooling='avg', 
                                          classes=number_of_classification_labels, 
                                          classifier_activation='sigmoid')
x = tf.keras.layers.Dense(units=number_of_classification_labels,
                          activation='sigmoid')(model.output)                                           
model = tf.keras.Model(inputs=model.input, outputs=x)

weights_filename = scripts_directory + "chexnet_antsxnet.h5"
if os.path.exists(weights_filename):
    model.load_weights(weights_filename)
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

###
#
# Set up the training generator
#

generator = batch_generator(batch_size=32,
                            demo=demo.loc[demo['fold'] == 'train'],
                            do_augmentation=True)

val_generator = batch_generator(batch_size=8,
                            demo=demo.loc[demo['fold'] == 'val'],
                            do_augmentation=True)

track = model.fit(x=generator, epochs=500, verbose=1, steps_per_epoch=100,
                  validation_data=val_generator, validation_steps=1,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='val_loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
          verbose=1, patience=10, mode='auto')
#       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001,
#          patience=20)
       ]
   )

model.save_weights(weights_filename)


