import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import glob

import random
import math

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/WMH/'
scripts_directory = base_directory + 'Scripts/SysuWithAttentionAndSite/'

from batch_slice_generator import batch_generator

template_size = (208, 208)
# template_size = (200, 200)
number_of_slices_per_image = 5
classes = (0, 1)

################################################
#
#  Create the model and load weights
#
################################################

number_of_classification_labels = len(classes)
image_modalities = ["T1"]
channel_size = len(image_modalities)

unet_model = antspynet.create_unet_model_2d((*template_size, channel_size),
   scalar_output_size=3, scalar_output_activation="softmax",
   number_of_outputs=1,
   mode="sigmoid", number_of_filters=(64, 96, 128, 256, 512),
   convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
   dropout_rate=0.0, weight_decay=0,
   additional_options=("attentionGating", "initialConvolutionKernelSize[5]"))
   

# unet_model = antspynet.create_sysu_media_unet_model_2d((*template_size, channel_size))

# unet_model = antspynet.create_unet_model_2d((*template_size, channel_size),
#     number_of_outputs=1, mode="sigmoid",
#     number_of_filters=(64, 96, 128, 256, 512), dropout_rate=0.0,
#     convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
#     weight_decay=1e-5,
#     additional_options=("nnUnetActivationStyle", "addAttentionGating", "initialConvolutionKernelSize[5]"))

weights_filename = scripts_directory + "wmhSliceSegmentationWeights_t1only.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

# unet_loss = antspynet.weighted_categorical_crossentropy((1, 10))

unet_loss = antspynet.binary_dice_coefficient(smoothing_factor=1.)
dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)

# unet_loss = antspynet.multilabel_surface_loss(dimensionality=2)
# dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=2, smoothing_factor=0.)
# surface_loss = antspynet.multilabel_surface_loss(dimensionality=2)

# def combined_loss(alpha):
#     def combined_loss_fixed(y_true, y_pred):
#         return (alpha * dice_loss(y_true, y_pred) +
#                 (1 - alpha) * surface_loss(y_true, y_pred))
#     return(combined_loss_fixed)
# wmh_loss = combined_loss(0.5)

unet_model.compile(optimizer=keras.optimizers.Adam(lr=2e-4),
                   loss=[unet_loss, "categorical_crossentropy"],
                   loss_weights=[1.0, 0.1],
                   metrics=[dice_loss])


################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_adni_images = glob.glob(base_directory + "ADNI_temp_flair_space_processed/*/*/*/*/*ants_FLAIRxT1.nii.gz")
t1_oasis3_images = glob.glob(base_directory + "FlairSpaceProcessed/*/*/NIFTI/*ants_FLAIRxT1.nii.gz")
t1_wmh2017_images = glob.glob(base_directory + "WMHChallenge2017/Nifti/*/*/pre/T1.nii.gz")

# t1_images = (*t1_adni_images, *t1_oasis3_images)
t1_images = t1_wmh2017_images

training_t1_files = list()
training_flair_files = list()
training_wmh_files = list()
training_seg_files = list()
training_site_ids = list()

for i in range(len(t1_images)):
    t1 = t1_images[i]

    subject_directory = os.path.dirname(t1)
#    flairs = glob.glob(subject_directory + "/*FLAIR_Preprocessed.nii.gz")
#    wmhs = glob.glob(subject_directory + "/*ants_FLAIR_combinedWMH.nii.gz")
#    segs = glob.glob(subject_directory + "/*ants_BrainSegmentation.nii.gz")

    flairs = glob.glob(subject_directory + "/*FLAIR.nii.gz")
    wmhs = glob.glob(subject_directory + "/../*wmh.nii.gz")
    segs = glob.glob(subject_directory + "/*deep_atropos.nii.gz")

    print(t1)
    if len(flairs) == 0:
        print("flair")
        continue

    if len(wmhs) == 0:
        print("wmh")
        continue

    if len(segs) == 0:
        print("segs")
        continue

    flair = flairs[0]
    wmh = wmhs[0]
    seg = segs[0]

    training_t1_files.append(t1)
    training_flair_files.append(flair)
    training_wmh_files.append(wmh)
    training_seg_files.append(seg)

    if "Amdterdam_GE3T" in t1:
        training_site_ids.append(0)
    elif "Singapore" in t1:
        training_site_ids.append(1)
    elif "Utrecht" in t1:
        training_site_ids.append(2)
    else:
        raise ValueError("No site id.")

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 30

# Split trainingData into "training" and "validation" componets for
# training the model.

number_of_data = len(training_t1_files)
sample_indices = random.sample(range(number_of_data), number_of_data)

generator = batch_generator(batch_size=batch_size,
                             image_size=template_size,
                             t1s=training_t1_files,
                             flairs=training_flair_files,
                             wmh_images=training_wmh_files,
                             segmentation_images=training_seg_files,
                             site_ids=training_site_ids,
                             use_t1s=True,
                             use_flairs=False,
                             use_segmentation_images=False,
                             number_of_slices_per_image=number_of_slices_per_image,
                             do_random_contralateral_flips=False,
                             do_histogram_intensity_warping=False,
                             do_simulate_bias_field=True,
                             do_add_noise=False,
                             do_data_augmentation=True,
                             use_rank_intensity_scaling=False
                            )

track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=50,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
          verbose=1, patience=10, mode='auto'),
       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001,
          patience=20)
       ]
   )

unet_model.save_weights(weights_filename)


