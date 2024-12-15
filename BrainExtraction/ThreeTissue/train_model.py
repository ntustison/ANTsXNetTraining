import ants
import antspynet

import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()

base_directory = '/home/ntustison/Data/BrainExtraction/'
scripts_directory = base_directory + 'Scripts/'

from batch_generator import batch_generator

template = ants.image_read(antspynet.get_antsxnet_data("nki"))
template_size = template.shape

################################################
#
#  Create the model and load weights
#
################################################

classes = ['background', 'brain', 'skull', 'misc']
number_of_classification_labels = len(classes)
image_modalities = ["T1"]
channel_size = len(image_modalities)

unet_model = antspynet.create_unet_model_3d((*template_size, channel_size),
    number_of_outputs=number_of_classification_labels,
    number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
    convolution_kernel_size=3, deconvolution_kernel_size=2,
    weight_decay=0.0)

weights_filename = scripts_directory + "brainExtraction.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.)
surface_loss = antspynet.multilabel_surface_loss()

def multilabel_combined_loss(alpha=0.5):
    def multilabel_combined_loss_fixed(y_true, y_pred):
        loss = (alpha * dice_loss(y_true, y_pred) + 
                (1-alpha) * surface_loss(y_true, y_pred)) 
        return(loss)
    return multilabel_combined_loss_fixed

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                   loss=multilabel_combined_loss(0.75),
                   metrics=dice_loss)

################################################
#
#  Load the brain data
#
################################################

print("Loading braindata.")

data_directory = base_directory + '/Data/'

training_image_files = list()
training_segmentation_files = glob.glob(data_directory + "**/*Mask.nii.gz", recursive=True)

if len(training_segmentation_files) == 0:
    print(data_directory)
    raise ValueError("No training images.")

for i in range(len(training_segmentation_files)):
    image_file = training_segmentation_files[i].replace("Mask.nii.gz", ".nii.gz")
    if not os.path.exists(image_file):
        raise ValueError("Template file doesn't exist.")
    training_image_files.append(image_file)

print("Total training image files: ", len(training_image_files))
print( "Training")

###
#
# Set up the training generator
#

batch_size = 4

generator = batch_generator(batch_size=batch_size,
                    template=template,
                    training_image_files=training_image_files,
                    training_segmentation_files=training_segmentation_files,
                    do_histogram_intensity_warping=True,
                    do_histogram_equalization=True,
                    do_histogram_rank_intensity=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    do_random_contralateral_flips=True,
                    do_resampling=True,
                    verbose=False)

track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor="loss",
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.75,
          verbose=1, patience=20, mode='auto'),
    #    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001,
    #       patience=20)
       ]
   )

unet_model.save_weights(weights_filename)


