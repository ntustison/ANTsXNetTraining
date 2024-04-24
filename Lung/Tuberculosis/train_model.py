import antspynet
import glob
import random

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

base_directory = '/home/ntustison/Data/Tuberculosis/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory 

image_size = (224, 224)

################################################
#
#  Create the model and load weights
#
################################################

weights_filename = scripts_directory + "tb_antsxnet.h5"

number_of_classification_labels = 14
number_of_channels = 3

chexnet_disease_categories = ['Atelectasis',
                              'Cardiomegaly',
                              'Effusion',
                              'Infiltration',
                              'Mass',
                              'Nodule',
                              'Pneumonia',
                              'Pneumothorax',
                              'Consolidation',
                              'Edema',
                              'Emphysema',
                              'Fibrosis',
                              'Pleural_Thickening',
                              'Hernia']

model = tf.keras.applications.DenseNet121(include_top=False, 
                                          weights="imagenet", 
                                          input_tensor=None, 
                                          input_shape=(*image_size, number_of_channels), 
                                          pooling='avg')
x = tf.keras.layers.Dense(units=len(chexnet_disease_categories),
                          activation='sigmoid')(model.output)                                           
model = tf.keras.Model(inputs=model.input, outputs=x)

if not os.path.exists(weights_filename):
    chexnet_weights = antspynet.get_pretrained_network("chexnetANTsXNetClassification")
    model.load_weights(chexnet_weights)

x = tf.keras.layers.Dense(units=1,
                          activation='sigmoid')(model.output)                                           
model = tf.keras.Model(inputs=model.input, outputs=x)

if os.path.exists(weights_filename):
    model.load_weights(weights_filename)

model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, decay=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

################################################
#
#  Get data
#
################################################

image_files = glob.glob(data_directory + "Images/TB_Chest_Radiography_Database/*/*.nii.gz")

training_image_files = list()
training_mask_files = list()
diagnoses = list()

for i in range(len(image_files)):
    image_file = image_files[i]
    mask_file = image_file.replace("Images", "Masks")
    if os.path.exists(image_file) and os.path.exists(mask_file):
        training_image_files.append(image_file)
        training_mask_files.append(mask_file)
        if "Normal" in os.path.basename(image_file):
            diagnoses.append(0)
        else:
            diagnoses.append(1)    

if len(diagnoses) == 0:
    raise ValueError("No training files.")

print("Number of training subjects: ", str(len(diagnoses)))

random_indices = list(range(len(diagnoses)))
random.shuffle(random_indices)

split_index = int(0.9*len(diagnoses))

###
#
# Set up the training generator
#

# generator = batch_generator(batch_size=32,
#                             image_files=training_image_files,
#                             mask_files=training_mask_files,
#                             diagnoses=diagnoses,
#                             do_augmentation=False)

# track = model.fit(x=generator, epochs=500, verbose=1, steps_per_epoch=100,
#     callbacks=[
#        keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
#            save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
#        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9,
#           verbose=1, patience=10, mode='auto')
#        ]
#    )

generator = batch_generator(batch_size=32,
                            image_files=training_image_files[:split_index],
                            mask_files=training_mask_files[:split_index],
                            diagnoses=diagnoses[:split_index],
                            do_augmentation=False)

val_generator = batch_generator(batch_size=8,
                            image_files=training_image_files[split_index:],
                            mask_files=training_mask_files[split_index:],
                            diagnoses=diagnoses[split_index:],
                            do_augmentation=False)

track = model.fit(x=generator, epochs=500, verbose=1, steps_per_epoch=100,
                  validation_data=val_generator, validation_steps=1,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='val_loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
          verbose=1, patience=10, mode='auto')
#       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001,
#          patience=20)
       ]
   )

model.save_weights(weights_filename)


