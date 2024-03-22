import ants
import antspynet

from tensorflow.keras.models import Model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"
import glob

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/EXACT09/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + 'Nifti/training/'

################################################
#
#  Create the model and load weights
#
################################################

patch_size = (160, 160, 160)
number_of_classification_labels = 1
channel_size = 1  # [CT, LeftLungMask, RightLungMask, AirwayMask]

unet_model = antspynet.create_unet_model_3d((*patch_size, channel_size),
   number_of_outputs=number_of_classification_labels, mode="sigmoid", 
   number_of_filters=(16, 32, 64, 128, 256),
   convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
   dropout_rate=0.0, weight_decay=0)

ce_loss = antspynet.weighted_categorical_crossentropy(weights=(1, 10))
dice_loss = antspynet.multilabel_dice_coefficient(smoothing_factor=0.)

weights_filename = scripts_directory + "weights_xx.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
                    loss=dice_loss,
                    metrics=[dice_loss])

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

ct_files = glob.glob(data_directory + "images/*.nii.gz")

training_ct_files = list()
training_seg_files = list()
training_label_files = list()
training_dist_files = list()

for i in range(len(ct_files)):

    ct_file = ct_files[i] 
    seg_file = ct_file.replace("images", "masks")
    label_file = ct_file.replace("images", "annotations")
    dist_file = ct_file.replace("images", "maurer")

    if(os.path.exists(ct_file) and os.path.exists(seg_file)) and os.path.exists(label_file):
        training_ct_files.append(ct_file)
        training_seg_files.append(seg_file)
        training_label_files.append(label_file)
        training_dist_files.append(dist_file)

print("Total training image files: ", len(training_ct_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 4

generator = batch_generator(batch_size=batch_size,
                    ct_files=training_ct_files,
                    seg_files=training_seg_files,
                    label_files=training_label_files,
                    dist_files=training_dist_files,
                    patch_size=patch_size,
                    number_of_channels=channel_size,
                    do_histogram_intensity_warping=False,
                    do_simulate_bias_field=False,
                    do_add_noise=False,
                    do_random_transformation=False)


track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95,
          verbose=1, patience=20, mode='auto'),
    #    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001,
    #       patience=20)
       ]
   )

unet_model.save_weights(weights_filename)


