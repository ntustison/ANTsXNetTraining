import ants
import antspynet

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from keras import Model
from keras.layers import Input

K.clear_session()

# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

################################################
#
#  Create the U-net model, losses, and load weights (if they exist)
#
################################################

number_of_priors = 0

template_priors = list()
for i in range(6):
    template_priors.append(ants.image_read("Oasis/priors" + str(i+1) + ".nii.gz"))
number_of_priors=len(template_priors)

channel_size = 1
image_size = (256, 256)

unet_model = antspynet.create_partial_convolution_unet_model_2d((*image_size, channel_size),
                                                                 number_of_priors=number_of_priors,
                                                                 number_of_filters=(32, 64, 128, 256, 512, 512),
                                                                 kernel_size=3,
                                                                 use_partial_conv=True)

weights_filename = None
if number_of_priors > 0:
    weights_filename = "t1_inpainting_with_priors_weights_round2.h5"
else:
    weights_filename = "t1_inpainting_weights.h5"
unet_model.load_weights(weights_filename)




# slice = ants.image_read(ants.get_ants_data("r16"))
# slice = antspynet.pad_or_crop_image_to_size(slice, image_size)
# slice = (slice - slice.min()) / (slice.max() - slice.min())
# mask = ants.image_read("r16roi.nii.gz")
# mask = mask * -1 + 1

slice_idx = 129
template = ants.image_read(antspynet.get_antsxnet_data("oasis"))
slice = ants.slice_image(template, idx=slice_idx, axis=1, collapse_strategy=1)
slice = antspynet.pad_or_crop_image_to_size(slice, image_size)
slice = (slice - slice.min()) / (slice.max() - slice.min())
mask = ants.image_read("r16roi3.nii.gz")
mask = mask * -1 + 1

# mask = ants.image_clone(slice) * 0 + 1

batchX = np.zeros((1, *image_size, 1))
batchX[0,:,:,0] = slice.numpy()
batchXMask = np.ones_like(batchX)
batchXMask[0,:,:,0] = np.squeeze(mask.numpy())


batchXPriors = np.zeros((1, *image_size, number_of_priors))
if number_of_priors > 0:
    for i in range(number_of_priors):
        slice = ants.slice_image(template_priors[i], idx=slice_idx, axis=1, collapse_strategy=1)
        slice = antspynet.pad_or_crop_image_to_size(slice, image_size)
        batchXPriors[0,:,:,i] = slice.numpy()

predicted_data = unet_model.predict([batchX, batchXMask, batchXPriors])

ants.image_write(ants.from_numpy(np.squeeze(batchX)), "slice.nii.gz")
ants.image_write(ants.from_numpy(np.squeeze(predicted_data)), "predicted.nii.gz")

