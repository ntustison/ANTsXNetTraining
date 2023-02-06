import ants
import antspynet

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

K.clear_session()

# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()


template = ants.image_read(antspynet.get_antsxnet_data("oasis"))

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

unet_model, input_mask = antspynet.create_partial_convolution_unet_model_2d((*image_size, channel_size),
                                                                            number_of_priors=0,
                                                                            number_of_filters=(16, 32, 64, 128),
                                                                            kernel_size=3,
                                                                            use_partial_conv=True)
weights_filename = "t1_pconv_identity_unet_weights.h5"

unet_model.load_weights(weights_filename)

slice_idx = 129

slice = ants.slice_image(template, axis=1, idx=slice_idx, collapse_strategy=1)
slice = antspynet.pad_or_crop_image_to_size(slice, image_size)
slice = (slice - slice.min()) / (slice.max() - slice.min())

batchX = np.zeros((1, *image_size, 1))
batchX[0,:,:,0] = slice.numpy()
batchXMask = np.ones_like(batchX)
predicted_data = unet_model.predict([batchX, batchXMask])

ants.image_write(ants.from_numpy(np.squeeze(batchX)), "slice.nii.gz")
ants.image_write(ants.from_numpy(np.squeeze(predicted_data)), "predicted.nii.gz")
