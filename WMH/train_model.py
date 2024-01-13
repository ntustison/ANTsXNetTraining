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
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/WMH/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + 'Nifti/'

################################################
#
#  Create the model and load weights
#
################################################

patch_size = (64, 64, 64)
number_of_classification_labels = 1
channel_size = 2

# unet_model = antspynet.create_unet_model_3d((*patch_size, channel_size),
#    number_of_outputs=number_of_classification_labels, mode="sigmoid", 
#    number_of_filters=(32, 64, 128, 256),
#    convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
#    dropout_rate=0.0, weight_decay=0)

# number_of_filters = (64, 96, 128, 256, 512)
number_of_filters = (64, 96, 128, 256, 512)

unet_model = antspynet.create_sysu_media_unet_model_3d((*patch_size, channel_size),
                                                       number_of_filters=number_of_filters)

# unet_loss = antspynet.multilabel_surface_loss(dimensionality=2)
# dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=2, smoothing_factor=0.)
binary_dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=1.0)
surface_loss = antspynet.binary_surface_loss()
wmh_loss = binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# dice_loss = antspynet.multilabel_dice_coefficient(smoothing_factor=0.0)
# categorical_loss = antspynet.weighted_categorical_crossentropy((0.25, 0.75))

# def combined_loss(alpha):
#     def combined_loss_fixed(y_true, y_pred):
#         return (alpha * binary_dice_loss(y_true, y_pred) +
#                 (1 - alpha) * surface_loss(y_true, y_pred))
#     return(combined_loss_fixed)
# wmh_loss = combined_loss(0.5)

# unet_model.compile(optimizer=keras.optimizers.Adam(lr=2e-4),
#                    loss=[unet_loss, "categorical_crossentropy"],
#                    loss_weights=[1.0, 0.1],
#                    metrics=[dice_loss])

weights_filename = scripts_directory + "wmh.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
                    loss=binary_dice_loss,
                    metrics=[binary_dice_loss, surface_loss])

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = glob.glob(data_directory + "*/T1.nii.gz")

training_t1_files = list()
training_t2_files = list()
training_atropos_files = list()
training_sysu_files = list()
training_bianca_files = list()

for i in range(len(t1_images)):

    subject_directory = os.path.dirname(t1_images[i])

    training_t1_files.append(t1_images[i])
    training_t2_files.append(t1_images[i].replace("T1", "T2_FLAIR"))
    training_atropos_files.append(t1_images[i].replace("T1", "atropos"))
    training_sysu_files.append(t1_images[i].replace("T1", "sysu"))
    training_bianca_files.append(t1_images[i].replace("T1", "bianca"))

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 16

generator = batch_generator(batch_size=batch_size,
                    t2_files=training_t2_files,
                    t1_files=training_t1_files,
                    atropos_files=training_atropos_files,
                    sysu_files=training_sysu_files,
                    bianca_files=training_bianca_files,
                    patch_size=patch_size,
                    number_of_channels=channel_size,
                    do_histogram_intensity_warping=True,
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


