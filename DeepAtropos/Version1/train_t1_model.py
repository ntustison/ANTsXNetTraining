import ants
import antspynet
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)

# tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/DeepAtroposHCP/'
scripts_directory = base_directory + 'Scripts/'

patch_size = (192, 224, 192)

data_directory = list()
# data_directory.append(base_directory + "Data/HCP-A/")
# data_directory.append(base_directory + "Data/HCP-YA/")
for i in [0, 1, 2]:
    data_directory.append(base_directory + "Data/HCP-A-" + str(i) + "/")
    data_directory.append(base_directory + "Data/HCP-YA-" + str(i) + "/")
data_directory.append(base_directory + "Data/HCP-sub-200001/")

template_t1 = list()
template_brain_mask = list()
template_segmentation = list()
template_priors = list()

hcp_template_inter_brain_segmentation = ants.image_read(antspynet.get_antsxnet_data("hcpinterTemplateBrainSegmentation"))

for i in range(len(data_directory)):
    template_t1.append(ants.image_read(data_directory[i] + "T_template0.nii.gz"))
    template_segmentation.append(ants.image_read(data_directory[i] + "T_templateBrainSegmentation_MT.nii.gz"))
    template_brain_mask.append(ants.threshold_image(template_segmentation[i], 0, 0, 0, 1))

template_priors = list()
for j in range(6):
    prior = ants.threshold_image(hcp_template_inter_brain_segmentation, j+1, j+1, 1, 0)
    prior_smooth = ants.smooth_image(prior, 1.0)
    template_priors.append(prior_smooth)

################################################
#
#  Create the model and load weights
#
################################################

classes = list(range(7))
number_of_classification_labels = len(classes)
image_modalities = ["T1"]
channel_size = len(image_modalities) + len(template_priors)

unet_model = antspynet.create_unet_model_3d((*patch_size, channel_size),
    number_of_outputs=number_of_classification_labels, mode="classification",
    number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
    convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
    weight_decay=0.0)

#########
#
#          DO NOT LOAD PREVIOUS WEIGHTS!!!!!!!!!!!!!!
#
#

ce_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 1.5, 1, 3, 4, 3, 3 ))
ce_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 1.5, 1, 3, 4, 3, 0.1 ))
ce_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 3, 1, 3, 4, 3, 0.1 ))
ce_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 10, 1, 3, 4, 3, 1 ))
ce_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 10, 1, 3, 4, 3, 0.5 )) # pretty good.  Just a tad too much WM relative to GM and DeepGM
ce_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 10, 1, 2, 4, 3, 0.5 ))
ce_loss = antspynet.weighted_categorical_crossentropy(weights=(0.05, 7.5, 1, 2, 4, 3, 0.5 ))

dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.)

def multilabel_combined_loss2(alpha=0.5):
    def multilabel_combined_loss_fixed(y_true, y_pred):
        loss = (alpha * dice_loss(y_true, y_pred) + 
                (1-alpha) * ce_loss(y_true, y_pred)) 
        return(loss)
    return multilabel_combined_loss_fixed

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
                   loss=multilabel_combined_loss2(0.1),
                   metrics=['accuracy', dice_loss])


weights_filename = scripts_directory + "DeepAtroposHcpT1Weights.h5"

if os.path.exists(weights_filename):
    print("Loading " + weights_filename)
    unet_model.load_weights(weights_filename)
else:
    unet_model_3 = antspynet.create_unet_model_3d((*patch_size, 3 + len(template_priors[0])),
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        weight_decay=0.0)
    unet_model_3.load_weights(scripts_directory + "DeepAtroposHcpT1T2FAWeights.h5")
    for i in range(len(unet_model.layers)):
        if i == 1:
           weights = unet_model_3.get_layer(index=i).get_weights()
           w0 = weights[0]
           weight = tf.convert_to_tensor(np.concatenate((w0[:,:,:,[0],:], w0[:,:,:,3:,:]), axis=-2))
           unet_model.layers[i].set_weights([weight, weights[1]])
        else:    
            print("Transferring weights layer " + str(i))
            unet_model.layers[i].set_weights(unet_model_3.get_layer(index=i).get_weights())


###
#
# Set up the training generator
#

batch_size = 4

generator = batch_generator(batch_size=batch_size,
                    template=template_t1,
                    priors=template_priors,
                    template_modalities=[template_t1],
                    template_segmentation=template_segmentation,
                    template_brain_mask=template_brain_mask,
                    patch_size=patch_size,
                    number_of_octants_per_image=4,
                    do_histogram_intensity_warping=True,
                    do_histogram_equalization=True,
                    do_histogram_rank_intensity=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    do_random_contralateral_flips=True,
                    do_resampling=True,
                    verbose=False)

track = unet_model.fit(x=generator, epochs=100, verbose=1, steps_per_epoch=32,
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


