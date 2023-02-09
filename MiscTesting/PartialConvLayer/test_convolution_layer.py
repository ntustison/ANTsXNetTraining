import ants
import antspynet
from antspynet import PartialConv2D

import numpy as np

from tensorflow import keras
from keras import Input, Model
from keras.layers import Conv2D, Input


image = ants.image_read(ants.get_ants_data("r16"))
mask = ants.image_read("~/Desktop/r16roi.nii.gz")

batchX = np.zeros((1, *image.shape, 1))
batchX[0,:,:,0] = image.numpy()
batchXMask = np.ones_like(batchX)
batchXMask[0,:,:,0] = np.squeeze(mask.numpy()) * -1 + 1

input_image = Input((*image.shape, 1))
input_mask = Input((*image.shape, 1))

do_pconv=True

if do_pconv:
    pconv_model = Model(inputs=[input_image, input_mask], outputs=conv)
    predicted_data = pconv_model.predict([batchX, batchXMask])
    ants.image_write(ants.from_numpy(np.squeeze(predicted_data)), "pconv_filters_r16.nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(pconv_model.layers[2].weights[0])), "pconv_weights.nii.gz")
    layer = pconv_model.layers[2]
    layer_model = Model(inputs=[input_image, input_mask], outputs=layer.output)
    predicted_data = layer_model.predict([batchX, batchXMask])
    ants.image_write(ants.from_numpy(np.squeeze(predicted_data[1])), "pconv_filters_image.nii.gz")


    # strides = (1,1)
    # output_img, output_mask1 = PartialConv2D(8, kernel_size=(7,7), strides=strides)([input_image, input_mask])
    # output_img, output_mask2 = PartialConv2D(16, kernel_size=(5,5), strides=strides)([output_img, output_mask1])
    # output_img, output_mask3 = PartialConv2D(32, kernel_size=(5,5), strides=strides)([output_img, output_mask2])
    # output_img, output_mask4 = PartialConv2D(64, kernel_size=(3,3), strides=strides)([output_img, output_mask3])
    # output_img, output_mask5 = PartialConv2D(64, kernel_size=(3,3), strides=strides)([output_img, output_mask4])
    # output_img, output_mask6 = PartialConv2D(64, kernel_size=(3,3), strides=strides)([output_img, output_mask5])
    # output_img, output_mask7 = PartialConv2D(64, kernel_size=(3,3), strides=strides)([output_img, output_mask6])
    # output_img, output_mask8 = PartialConv2D(64, kernel_size=(3,3), strides=strides)([output_img, output_mask7])
    # model = Model(
    #   inputs=[input_image, input_mask],
    #   outputs=[
    #       output_img, output_mask1, output_mask2,
    #       output_mask3, output_mask4, output_mask5,
    #       output_mask6, output_mask7, output_mask8
    #   ])
    # output_img, o1, o2, o3, o4, o5, o6, o7, o8 = model.predict([batchX, batchXMask])
    # ants.image_write(ants.from_numpy(np.squeeze(o1)), "o1.nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(o2)), "o2.nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(o3)), "o3.nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(o4)), "o4.nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(o5)), "o5.nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(o6)), "o6.nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(o7)), "o7.nii.gz")
else:
    conv = Conv2D(filters=8,
                  kernel_size=3,
                  padding="same")(input_image)
    conv_model = Model(inputs=input_image, outputs=conv)
    predicted_data = conv_model.predict(batchX)
    ants.image_write(ants.from_numpy(np.squeeze(predicted_data)), "conv_filters_r16.nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(conv_model.layers[1].weights[0])), "conv_weights.nii.gz")
