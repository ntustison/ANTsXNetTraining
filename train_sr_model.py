import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import glob

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda

K.clear_session()
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     pass

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

base_directory = '/home/ntustison/Data/Allen/'
scripts_directory = base_directory + 'Scripts/'

from batch_sr_generator import batch_generator

################################################
#
#  Create the VGG model and load weights
#
################################################

vgg16_model = antspynet.create_vgg_model_2d((224, 224, 3),
                                             number_of_classification_labels=1,
                                             layers=(1, 2, 3, 4, 4),
                                             lowest_resolution=64,
                                             convolution_kernel_size=(3, 3),
                                             pool_size=(2, 2),
                                             strides=(2, 2),
                                             number_of_dense_units=4096,
                                             dropout_rate=0.0,
                                             style=16,
                                             mode='classification')

vgg16_keras = keras.applications.VGG16(include_top=True,
                                       weights='imagenet',
                                       input_tensor=None,
                                       input_shape=None,
                                       pooling=None,
                                       classes=1000,
                                       classifier_activation='softmax')
for i in range(len(vgg16_keras.layers)-1):
    vgg16_model.layers[i].set_weights(vgg16_keras.layers[i].get_weights())

vgg16_topless_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.layers[18].output)
vgg16_topless_model.layers[0]._batch_input_shape = (None, None, None, 3)
vgg16_topless_model = keras.models.model_from_json(vgg16_topless_model.to_json())

max_pool_layers = [3, 6, 10]
vgg16_topless_model.outputs = [vgg16_topless_model.layers[i].output for i in max_pool_layers]
vgg16_topless_model = Model(inputs=vgg16_topless_model.inputs, outputs=vgg16_topless_model.outputs)

image_size = (512, 512, 3)
vgg_input_image = Input(shape=image_size)

# Scaling for VGG input
vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]
processed = Lambda(lambda x: (x - vgg_mean) / vgg_std)(vgg_input_image)
vgg16_model = Model(inputs=vgg_input_image, outputs=vgg16_topless_model(processed))
# vgg16_model = Model(inputs=vgg_input_image, outputs=vgg16_topless_model(vgg_input_image))
vgg16_model.trainable = False
vgg16_model.compile(loss='mse', optimizer='adam')

################################################
#
#  Create the model and load weights
#
################################################

input_image_size = (256, 256, 3)

sr_model = antspynet.create_deep_back_projection_network_model_2d(input_image_size,
   number_of_outputs=3, convolution_kernel_size=(6, 6), strides=(2, 2))

weights_filename = scripts_directory + "/allen_sr_weights.h5"
if os.path.exists(weights_filename):
    sr_model.load_weights(weights_filename)

# loss_total = tf.keras.losses.MeanSquaredError()

def combined_loss(weights):

    def l1_norm(y_true, y_pred):
        if len(y_true.shape) == 4:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
        elif len(y_true.shape) == 3:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

    def gram_matrix(x):
        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"
        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)
        # Normalize with channels, height and width
        gram = gram /  K.cast(C * H * W, x.dtype)
        return gram

    def loss_style(output, vgg_true):
        """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
        loss = 0
        for o, g in zip(output, vgg_true):
            loss += l1_norm(gram_matrix(o), gram_matrix(g))
        return loss

    def loss_tv(y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""
        # Calculate total variation loss
        a = l1_norm(y_comp[:,1:,:,:], y_comp[:,:-1,:,:])
        b = l1_norm(y_comp[:,:,1:,:], y_comp[:,:,:-1,:])
        return a+b

    def combined_loss_fixed(y_true, y_pred):

        # Compute the vgg features.
        vgg_pred = vgg16_model(y_pred)
        vgg_true = vgg16_model(y_true)
        
        #Compute loss components
        l1 = K.cast(l1_norm(y_true, y_pred), 'float32')
        l2 = K.cast(loss_style(vgg_pred, vgg_true), 'float32')
        l3 = K.cast(loss_tv(y_pred), 'float32')

        # Return loss function
        return tf.math.reduce_mean(weights[0] * l1 + weights[1] * l2 + weights[2] * l3)
        
    return combined_loss_fixed

loss_total = combined_loss((1.0, 5.0, 0.0))

sr_model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4), 
                 loss=loss_total)

################################################
#
#  Load the data
#
################################################

print("Loading hist data.")

sr_files = glob.glob(base_directory + "SuperResolution/*sr.nii.gz")
lr_files = list()

count = 0
for i in range(len(sr_files)):
    lr_file = sr_files[i].replace("sr.nii.gz", "lr.nii.gz")
    if not os.path.exists(lr_file):
        raise ValueError(lr_file + " does not exist.")

    lr_files.append(lr_file)
    count = count + 1
    # if count >= 1000:
    #     break

print("Total training image files: ", len(sr_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 2

generator = batch_generator(batch_size=batch_size,
                             lr_images=lr_files,
                             sr_images=sr_files,
                             do_smoothing=True,
                             do_random_contralateral_flips=True,
                             do_simulate_bias_field=False,
                             do_histogram_intensity_warping=False,
                             do_add_noise=False
                            )


track = sr_model.fit(x=generator, epochs=400, verbose=1, steps_per_epoch=256,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
          verbose=1, patience=10, mode='auto'),
       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000000001,
          patience=20)
       ]
   )

sr_model.save_weights(weights_filename)

