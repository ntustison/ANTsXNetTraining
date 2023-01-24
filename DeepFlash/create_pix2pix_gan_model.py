import ants
import antspynet

import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Conv3D, LeakyReLU

import numpy as np
import os

class Pix2PixGanModel(object):
    """
    Pix2Pix GAN model

    Ported from:

    https://github.com/eriklindernoren/Keras-GAN/tree/master/pix2pix

    Arguments
    ---------
    input_image_size : tuple
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    Returns
    -------
    """

    def __init__(self, input_image_size):

        super(Pix2PixGanModel, self).__init__()

        self.input_image_size = input_image_size
        self.number_of_channels = self.input_image_size[-1]

        self.dimensionality = 3
        if len(self.input_image_size) == 3:
            self.dimensionality = 2
        elif len(self.input_image_size) == 4:
            self.dimensionality = 3
        else:
            raise ValueError("Incorrect size for input_image_size.")

        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

        # Build discriminator

        self.discriminator_patch_size = None
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer, metrics=['acc'])
        self.discriminator.trainable = False

        # Build u-net like generators

        self.generator = self.build_generator()

        image_source = Input(shape=input_image_size)
        image_target = Input(shape=input_image_size)

        fake_image_target = self.generator(image_source)

        # Check images

        validity = self.discriminator([fake_image_target, image_source])

        # Combined models

        self.combined_model = Model(inputs=[image_source, image_target],
                                    outputs=[validity, fake_image_target])
        self.combined_model.compile(loss=['mse', 'mae'],
                                    loss_weights=[1.0, 100.0],
                                    optimizer=optimizer)

    def build_generator(self):

        unet_model = None

        if self.dimensionality == 2:
            unet_model = antspynet.create_unet_model_2d(self.input_image_size,
                number_of_outputs=1, number_of_filters_at_base_layer=32,
                number_of_layers=4, mode="regression")
        else:
            unet_model = antspynet.create_unet_model_3d(self.input_image_size,
                number_of_outputs=1, number_of_filters_at_base_layer=32,
                number_of_layers=4, mode="regression")

        return(unet_model)

    def build_discriminator(self):

        def build_layer(input, number_of_filters, kernel_size=4):
            layer = input
            if self.dimensionality == 2:
                layer = Conv2D(filters=number_of_filters,
                                 kernel_size=kernel_size,
                                 strides=2,
                                 padding='same')(layer)
            else:
                layer = Conv3D(filters=number_of_filters,
                                 kernel_size=kernel_size,
                                 strides=2,
                                 padding='same')(layer)
            layer = LeakyReLU(alpha=0.2)(layer)
            return(layer)

        image_source = Input(shape=self.input_image_size)
        image_target = Input(self.input_image_size)

        image_combined = Concatenate(axis=-1)([image_source, image_target])

        layers = list()
        layers.append(build_layer(image_combined, 32))
        layers.append(build_layer(layers[0], 64))
        layers.append(build_layer(layers[1], 128))
        layers.append(build_layer(layers[2], 256))

        validity = None
        if self.dimensionality == 2:
            validity = Conv2D(filters=1,
                              kernel_size=4,
                              strides=1,
                              padding='same')(layers[3])
        else:
            validity = Conv3D(filters=1,
                              kernel_size=4,
                              strides=1,
                              padding='same')(layers[3])

        if self.discriminator_patch_size is None:
            self.discriminator_patch_size = K.int_shape(validity)[1:]

        discriminator = Model(inputs=[image_source, image_target], outputs=validity)
        return(discriminator)

    def train(self, X_source_images, X_target_images, number_of_epochs, batch_size=128,
              sample_interval=None, output_prefix='./output'):

        if not os.path.exists(os.path.dirname(output_prefix)):
            os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

        valid = np.ones((batch_size, *self.discriminator_patch_size))
        fake = np.zeros((batch_size, *self.discriminator_patch_size))

        best_g_loss = np.Inf

        for epoch in range(number_of_epochs):

            indices_source = np.random.randint(0, X_source_images.shape[0] - 1, batch_size)
            images_source = X_source_images[indices_source]

            indices_target = np.random.randint(0, X_target_images.shape[0] - 1, batch_size)
            images_target = X_target_images[indices_target]

            # train discriminator

            fake_images_target = self.generator.predict(images_source)

            loss_fake = self.discriminator.train_on_batch([images_source, fake_images_target], fake)
            loss_real = self.discriminator.train_on_batch([images_source, images_target], valid)
            d_loss = 0.5 * np.add(loss_real, loss_fake)

            d_loss = list()
            for i in range(len(loss_real)):
                d_loss.append(0.5 * (loss_real[i] + loss_fake[i]))

            # train generator

            g_loss = self.combined_model.train_on_batch([images_source, images_target], [valid, images_target])

            # Save weights

            if best_g_loss < g_loss[0]:
                self.generator.save_weights(output_prefix + "GeneratorWeights.h5")
                self.discriminator.save_weights(output_prefix + "DiscriminatorWeights.h5")
                best_g_loss = g_loss[0]

            print("Epoch ", epoch, ": [Discriminator loss: ", d_loss, "] ", "[Generator loss: ", g_loss, "]")

            if sample_interval is not None:
                if epoch % sample_interval == 0:

                    random_index = np.random.randint(0, X_source_images.shape[0] - 1, 1)

                    if self.dimensionality == 2:
                       random_array = X_source_images[random_index,:,:,:]
                    else:
                       random_array = X_source_images[random_index,:,:,:,:]

                    generated_array = self.generator.predict(random_array)

                    random_image = ants.from_numpy(np.squeeze(random_array))
                    generated_image = ants.from_numpy(np.squeeze(generated_array))

                    ants.image_write(random_image, output_prefix + "OriginalImage" + str(epoch) + ".nii.gz")
                    ants.image_write(generated_image, output_prefix + "GeneratedImage" + str(epoch) + ".nii.gz")



