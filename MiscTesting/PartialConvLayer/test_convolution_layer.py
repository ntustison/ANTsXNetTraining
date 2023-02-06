import ants
import antspynet
from antspynet import PartialConv2D

import numpy as np

from tensorflow import keras
from keras import Input, Model
from keras.layers import Conv2D, Input


image = ants.image_read(ants.get_ants_data("r16"))

batchX = np.zeros((1, *image.shape, 1))
batchX[0,:,:,0] = image.numpy()
batchXMask = np.ones_like(batchX)

input_image = Input((*image.shape, 1))
input_mask = Input((*image.shape, 1))

do_pconv=False

if do_pconv:
    conv, mask = PartialConv2D(filters=8,
                               kernel_size=3,
                               padding="same")([input_image, input_mask])
    pconv_model = Model(inputs=[input_image, input_mask], outputs=conv)
    predicted_data = pconv_model.predict([batchX, batchXMask])
    ants.image_write(ants.from_numpy(np.squeeze(predicted_data)), "pconv_filters_r16.nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(pconv_model.layers[2].weights[0])), "pconv_weights.nii.gz")
else:
    conv = Conv2D(filters=8,
                  kernel_size=3,
                  padding="same")(input_image)
    conv_model = Model(inputs=input_image, outputs=conv)
    predicted_data = conv_model.predict(batchX)
    ants.image_write(ants.from_numpy(np.squeeze(predicted_data)), "conv_filters_r16.nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(conv_model.layers[1].weights[0])), "conv_weights.nii.gz")
