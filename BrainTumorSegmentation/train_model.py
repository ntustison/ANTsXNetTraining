import ants
import antspynet

from tensorflow.keras.models import Model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

base_directory = '/home/ntustison/Data/BRATS/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + 'TCIA/'

################################################
#
#  Create the model and load weights
#
################################################

patch_size = (64, 64, 64)
number_of_classification_labels = 5
channel_size = 5  # [FLAIR, T1, T1GD, T2, MASK]

unet_model = antspynet.create_unet_model_3d((*patch_size, channel_size),
   number_of_outputs=number_of_classification_labels, mode="sigmoid", 
   number_of_filters=(32, 64, 128, 256, 512),
   convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
   dropout_rate=0.0, weight_decay=0)

# number_of_filters = (64, 96, 128, 256, 512)
number_of_filters = (64, 96, 128, 256, 512)

dice_loss = antspynet.multilabel_dice_coefficient(dimensionality = 3, smoothing_factor=0.0)
categorical_loss = antspynet.weighted_categorical_crossentropy((1, 10, 10, 10, 10))

weights_filename = scripts_directory + "brats_stage2.h5"
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

t1_files = glob.glob(data_directory + "images_structural_unstripped/*/*_T1_*.nii.gz")

training_t1_files = list()
training_t2_files = list()
training_flair_files = list()
training_t1gd_files = list()
training_brain_mask_files = list()
training_seg_files = list()

for i in range(len(t1_files)):

    t1_file = t1_files[i] 
    t2_file = t1_file.replace("T1", "T2")
    t1gd_file = t1_file.replace("T1", "T1GD")
    flair_file = t1_file.replace("T1", "FLAIR") 

    mask_file = t1_file.replace("images_structural_unstripped", "ants_brain_extraction")
    mask_file = mask_file.replace("T1_unstripped", "T1_brain_mask")

    seg_file = t1_file.replace("images_structural_unstripped", "automated_segm")
    seg_file = seg_file.replace("T1_unstripped", "automated_approx_segm")
    seg_file = os.path.normpath(os.path.dirname(seg_file) + "/../") + "/" + os.path.basename(seg_file)

    if(os.path.exists(t1_file) and 
       os.path.exists(t2_file) and
       os.path.exists(t1gd_file) and
       os.path.exists(flair_file) and
       os.path.exists(mask_file) and
       os.path.exists(seg_file)):

        training_t1_files.append(t1_file)
        training_t2_files.append(t2_file)
        training_t1gd_files.append(t1gd_file)
        training_flair_files.append(flair_file)
        training_brain_mask_files.append(mask_file)
        training_seg_files.append(seg_file)

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 16

generator = batch_generator(batch_size=batch_size,
                    flair_files=training_flair_files,
                    t1_files=training_t1_files,
                    t1gd_files=training_t1gd_files,              
                    t2_files=training_t2_files,
                    brain_mask_files=training_brain_mask_files,
                    seg_files=training_seg_files,
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


