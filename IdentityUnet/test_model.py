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

channel_size = 1
image_size = (256, 256)

# unet_model = antspynet.create_unet_model_2d((*image_size, channel_size),
#                                             number_of_outputs=1,
#                                             number_of_filters=(16, 32, 64, 128),
#                                             mode="sigmoid",
#                                             dropout_rate=0.0,
#                                             convolution_kernel_size=3,
#                                             deconvolution_kernel_size=2,
#                                             weight_decay=0.0)
# weights_filename = "t1_identity_unet_weights.h5"

use_partial_conv=True

unet_model = antspynet.create_partial_convolution_unet_model_2d((*image_size, channel_size),
                                                                 number_of_priors=0,
                                                                 number_of_filters=(16, 32, 64, 128),
                                                                 kernel_size=3,
                                                                 use_partial_conv=use_partial_conv)

if use_partial_conv:
    weights_filename = "t1_pconv_identity_unet_weights.h5"
else:
    weights_filename = "t1_no_pconv_identity_unet_weights.h5"
unet_model.load_weights(weights_filename)

slice = ants.image_read(ants.get_ants_data("r16"))
slice = antspynet.pad_or_crop_image_to_size(slice, image_size)
slice = (slice - slice.min()) / (slice.max() - slice.min())

mask = ants.image_read("/Users/ntustison/Desktop/r16roi.nii.gz") * -1 + 1


batchX = np.zeros((1, *image_size, 1))
batchX[0,:,:,0] = slice.numpy()
batchXMask = np.ones_like(batchX)
# batchXMask[0,:,:,0] = np.squeeze(mask.numpy())

predicted_data = unet_model.predict([batchX, batchXMask])

ants.image_write(ants.from_numpy(np.squeeze(batchX)), "slice.nii.gz")
ants.image_write(ants.from_numpy(np.squeeze(predicted_data)), "predicted.nii.gz")

input_image = Input((*image_size, 1))
input_mask = Input((*image_size, 1))

layer = unet_model.get_layer('partial_conv2d_3')
layer_model = Model(inputs=unet_model.input, outputs=layer.output)
predicted_data = layer_model.predict([batchX, batchXMask])
ants.image_write(ants.from_numpy(np.squeeze(predicted_data[0])), "predicted_data0.nii.gz")
ants.image_write(ants.from_numpy(np.squeeze(predicted_data[1])), "predicted_data1.nii.gz")
