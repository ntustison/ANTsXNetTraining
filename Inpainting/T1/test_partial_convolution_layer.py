import ants
import antspynet

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras import Model


mask = ants.image_read("r16mask.nii.gz")

image_size = mask.shape
number_of_priors = 6
input_size = (image_size[0], image_size[1], 7)

formatted_img = np.ones((1, *input_size))
formatted_mask = np.zeros((1, *input_size))
formatted_mask[0,:,:,0] = np.squeeze(mask.numpy())
formatted_mask = 1 - formatted_mask

ants.plot(ants.from_numpy(formatted_mask[0,:,:,0]))

input_img = Input(input_size)
input_mask = Input(input_size)
output_img, output_mask1 = antspynet.PartialConv2D(8, kernel_size=(7,7), strides=(2,2))([input_img, input_mask])
output_img, output_mask2 = antspynet.PartialConv2D(16, kernel_size=(5,5), strides=(2,2))([output_img, output_mask1])
output_img, output_mask3 = antspynet.PartialConv2D(32, kernel_size=(5,5), strides=(2,2))([output_img, output_mask2])
output_img, output_mask4 = antspynet.PartialConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask3])
output_img, output_mask5 = antspynet.PartialConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask4])
output_img, output_mask6 = antspynet.PartialConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask5])
output_img, output_mask7 = antspynet.PartialConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask6])
output_img, output_mask8 = antspynet.PartialConv2D(64, kernel_size=(3,3), strides=(2,2))([output_img, output_mask7])

# Create model
model = Model(
    inputs=[input_img, input_mask],
    outputs=[
        output_img, output_mask1, output_mask2,
        output_mask3, output_mask4, output_mask5,
        output_mask6, output_mask7, output_mask8
    ])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

output_img, o1, o2, o3, o4, o5, o6, o7, o8 = model.predict([formatted_img, formatted_mask])

ants.plot(ants.from_numpy(formatted_mask[0,:,:,0]))