import ants
import antspynet
import numpy as np
import glob

import os
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from batch_combined_generator import batch_generator

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)

# tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/HoaLabeling/'
scripts_directory = base_directory + 'Scripts/'
priors_directory = base_directory + "Data/PriorProbabilityImages/"
labels_directory = base_directory + "Data/SubcorticalParcellations/dseg2/"
crop_size = (160, 176, 160)

labels = None
batch_size = None
batch_size = 1  

template_t1_files = list()
template_segmentation_files = list()

def reshape_image(image, interp_type = "linear", crop_size=crop_size):
    image_resampled = None
    if interp_type == "linear":
        image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=0)
    else:        
        image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=1)
    image_cropped = antspynet.pad_or_crop_image_to_size(image_resampled, crop_size)
    return image_cropped

template = reshape_image(ants.image_read(priors_directory + "prior1.nii.gz"))

t1_files = glob.glob(base_directory + "Data/T1w*/*T1w_extracted.nii.gz")

for i in range(len(t1_files)):
    t1_file = t1_files[i]
    labels_file = t1_file.replace("T1w_extracted.nii.gz", "dseg2.nii.gz")
    template_t1_files.append(t1_file)
    template_segmentation_files.append(labels_file)

print("Training data size: " + str(len(template_t1_files)))

################################################
#
#  Create the model and load weights
#
################################################

hoa_lateral_labels = (0, 3, 4, 5, 6, 15, 24)
hoa_lateral_left_labels = (1, 7, 9, 11, 13, 16, 18, 20, 22, 25, 27, 29, 31)
hoa_lateral_right_labels = (2, 8, 10, 12, 14, 17, 19, 21, 23, 26, 28, 30, 32)
hoa_extra_labels = (33, 34, 35)

labels = sorted((*hoa_lateral_labels, *hoa_lateral_left_labels, *hoa_extra_labels))
print("Labels: ", labels)
number_of_classification_labels = len(labels)
image_modalities = ["T1"]
channel_size = len(image_modalities)

unet_model = antspynet.create_unet_model_3d((*template.shape, channel_size),
    number_of_outputs=number_of_classification_labels, mode="classification",
    number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
    convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
    weight_decay=0.0)

weights = [1.0] * number_of_classification_labels
weights[0] = 0.001

ce_loss = antspynet.weighted_categorical_crossentropy(weights=tuple(weights))
# ce_loss = tf.keras.losses.CategoricalCrossentropy()

dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.)
surface_loss = antspynet.multilabel_surface_loss()

binary_dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.)
binary_surface_loss = antspynet.binary_surface_loss()

def multilabel_combined_loss(alpha=0.5):
    def multilabel_combined_loss_fixed(y_true, y_pred):
        loss = (alpha * dice_loss(y_true, y_pred) + 
                (1-alpha) * surface_loss(y_true, y_pred)) 
        return(loss)
    return multilabel_combined_loss_fixed


# multiple outputs

penultimate_layer = unet_model.layers[-2].output

# whole
output2 = Conv3D(filters=1,
                 kernel_size=(1, 1, 1),
                 activation='sigmoid',
                 kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

unet2_model = Model(inputs=unet_model.input, outputs=[unet_model.output, output2])

weights_filename = scripts_directory + "HoaCombinedNoPriorMultipleOutputsWeights.h5"
if os.path.exists(weights_filename):
    print("Loading " + weights_filename)
    unet2_model.load_weights(weights_filename)

unet2_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
                    loss=[dice_loss, binary_dice_loss],
                    loss_weights=[0.75, 0.25],
                    metrics=[dice_loss, None])

###
#
# Set up the training generator
#

generator = batch_generator(batch_size=batch_size,
                    template=template,
                    crop_size=crop_size,
                    prior_files=[],
                    training_image_files=template_t1_files,
                    training_segmentation_files=template_segmentation_files,
                    classification_labels=labels,
                    do_histogram_intensity_warping=True,
                    do_histogram_equalization=False,
                    do_histogram_rank_intensity=False,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    do_random_contralateral_flips=True,
                    do_resampling=True,
                    use_multiple_outputs=True,
                    verbose=False)

track = unet2_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.CSVLogger(scripts_directory + "no_priors_model_log.csv", append=True, separator=';'), 
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95,
          verbose=1, patience=20, mode='auto'),
    #    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001,
    #       patience=20)
       ]
   )

unet2_model.save_weights(weights_filename)


