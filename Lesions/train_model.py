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

base_directory = '/home/ntustison/Data/Lesions/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + 'ATLAS_2/'

################################################
#
#  Create the model and load weights
#
################################################

patch_size = (64, 64, 64)
number_of_classification_labels = 2
channel_size = 1


number_of_filters = (64, 96, 128, 256, 512)
unet_model = antspynet.create_sysu_media_unet_model_3d((*patch_size, channel_size),
                                                       number_of_filters=number_of_filters)

dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.0)
ce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

weights_filename = scripts_directory + "lesion_patch.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)
else:
    brats_unet_model = antspynet.create_sysu_media_unet_model_3d((*patch_size, 4),
                                number_of_filters=number_of_filters)
    brats_weights = antspynet.get_pretrained_network("bratsStage1")
    brats_unet_model.load_weights(brats_weights)
    for i in range(3, len(unet_model.layers)):
        print("Transferring layer " + str(i))
        unet_model.get_layer(index=i).set_weights(brats_unet_model.get_layer(index=i).get_weights())

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
                    loss=dice_loss,
                    metrics=[dice_loss])

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_files = glob.glob(data_directory + "Training/R*/sub*/ses*/anat/*T1w.nii.gz")

training_t1_files = list()
training_lesion_mask_files = list()
training_brain_mask_files = list()

for i in range(len(t1_files)):

    t1_file = t1_files[i] 
    lesion_mask_file = t1_file.replace("T1w.nii.gz", "label-L_desc-T1lesion_mask.nii.gz")
    brain_mask_file = t1_file.replace("T1w.nii.gz", "T1w_brain_mask.nii.gz")

    if (os.path.exists(t1_file) and 
        os.path.exists(lesion_mask_file) and
        os.path.exists(brain_mask_file)):

        training_t1_files.append(t1_file)
        training_lesion_mask_files.append(lesion_mask_file)
        training_brain_mask_files.append(brain_mask_file)
    else:
        print( "----> " + t1_file)    
        print( "      " + lesion_mask_file)    
        print( "      " + brain_mask_file)    

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 16

generator = batch_generator(batch_size=batch_size,
                    t1_files=training_t1_files,
                    lesion_mask_files=training_lesion_mask_files,              
                    brain_mask_files=training_brain_mask_files,
                    patch_size=patch_size,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True)

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


