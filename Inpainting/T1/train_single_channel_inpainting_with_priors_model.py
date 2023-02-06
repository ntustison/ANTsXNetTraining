import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob
import pandas as pd
import numpy as np
import time

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate

from batch_inpainting_generator import batch_generator

K.clear_session()

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

base_directory = '/home/ntustison/Data/Inpainting/'
scripts_directory = base_directory + 'T1/'
template_directory = base_directory + "Oasis/"

template = ants.image_read(antspynet.get_antsxnet_data("oasis"))
template_labels = ants.image_read(template_directory + "dktWithWhiteMatterLobes.nii.gz")
template_roi = ants.image_read(template_directory + "brainMaskDilated.nii.gz")
template_priors = list()
for i in range(6):
    template_priors.append(ants.image_read(template_directory + "priors" + str(i+1) + ".nii.gz"))

################################################
#
#  Create the VGG model and load weights
#
################################################

# vgg_mean = [0.485, 0.456, 0.406]
# vgg_std = [0.229, 0.224, 0.225]

# input_image = Input(shape=image_size)
# vgg16 = keras.applications.VGG16(weights='imagenet', include_top=False)
# vgg_layers = [3, 6, 10]
# vgg16.outputs = [vgg16.layers[i].output for i in vgg_layers]
# processed = Lambda(lambda x: (x-vgg_mean) / vgg_std)(input_image)
# vgg16_model = Model(inputs=input_image, outputs=vgg16(processed))
# vgg16_model.trainable = False
# vgg16_model.compile(loss='mse', optimizer='adam')

image_modalities = ["T1"]
number_of_channels = len(image_modalities)
template_size = template.shape

vgg16_top_model = antspynet.create_vgg_model_2d((224, 224, number_of_channels),
                                             number_of_classification_labels=1,
                                             layers=(1, 2, 3, 4, 4),
                                             lowest_resolution=64,
                                             convolution_kernel_size=(3, 3),
                                             pool_size=(2, 2),
                                             strides=(2, 2),
                                             number_of_dense_units=4096,
                                             dropout_rate=0.0,
                                             style=16,
                                             mode='regression')

vgg16_topless_model = Model(inputs=vgg16_top_model.inputs, outputs=vgg16_top_model.layers[18].output)
vgg16_topless_model.layers[0]._batch_input_shape = (None, None, None, 1)
vgg16_topless_model = keras.models.model_from_json(vgg16_topless_model.to_json())

vgg_weights_filename = scripts_directory + "vgg16_imagenet_single_channel_t1brain_weights.h5"
if os.path.exists(vgg_weights_filename):
    vgg16_top_model.load_weights(vgg_weights_filename)
    for i in range(len(vgg16_topless_model.layers)):
        vgg16_topless_model.layers[i].set_weights(vgg16_top_model.layers[i].get_weights())
else:
    raise ValueError("Weights file does not exist.")

max_pool_layers = [3, 6, 10]
vgg16_topless_model.outputs = [vgg16_topless_model.layers[i].output for i in max_pool_layers]
vgg16_topless_model = Model(inputs=vgg16_topless_model.inputs, outputs=vgg16_topless_model.outputs)
# vgg16_topless_model = keras.models.model_from_json(vgg16_topless_model.to_json())

image_size = (256, 256, number_of_channels)
vgg_input_image = Input(shape=image_size)

# Scaling for VGG input
# vgg_mean = [0.485, 0.456, 0.406]
# vgg_std = [0.229, 0.224, 0.225]
# processed = Lambda(lambda x: (x - vgg_mean) / vgg_std)(vgg_input_image)
# vgg16_model = Model(inputs=vgg_input_image, outputs=vgg16_topless_model(processed))

vgg16_model = Model(inputs=vgg_input_image, outputs=vgg16_topless_model(vgg_input_image))
vgg16_model.trainable = False
vgg16_model.compile(loss='mse', optimizer='adam')

################################################
#
#  Create the U-net model, losses, and load weights (if they exist)
#
################################################

image_size = (256, 256, number_of_channels)
print("Unet model with " + str(len(template_priors)) + " priors.")
inpainting_unet, input_mask = antspynet.create_partial_convolution_unet_model_2d(image_size,
                                                                                 number_of_priors=len(template_priors),
                                                                                 number_of_filters=(32, 64, 128, 256, 512),
                                                                                 kernel_size=3)

def loss_total(x_mask):

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

    def loss_hole(mask, y_true, y_pred):
        return l1_norm((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(mask, y_true, y_pred):
        return l1_norm(mask * y_true, mask * y_pred)

    def loss_perceptual(vgg_out, vgg_gt, vgg_comp):
        """Perceptual loss based on VGG16, see. eq. 3 in paper"""
        loss = 0
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += l1_norm(o, g) + l1_norm(c, g)
        return loss

    def loss_style(output, vgg_gt):
        """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
        loss = 0
        for o, g in zip(output, vgg_gt):
            loss += l1_norm(gram_matrix(o), gram_matrix(g))
        return loss

    def loss_tv(mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""
        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.conv2d(K.cast(1-mask, 'float32'), kernel, data_format='channels_last', padding='same')
        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        P = K.cast(K.greater(dilated_mask, 0), 'float32') * y_comp
        # Calculate total variation loss
        a = l1_norm(P[:,1:,:,:], P[:,:-1,:,:])
        b = l1_norm(P[:,:,1:,:], P[:,:,:-1,:])
        return a+b

    """
    Creates a loss function which sums all the loss components
    and multiplies by their weights. See paper eq. 7.
    """
    def loss(y_true, y_pred):

        # Compute predicted image with non-hole pixels set to ground truth
        y_comp = (1.0-x_mask) * y_pred + x_mask * y_true

        # Compute the vgg features.
        vgg_out = vgg16_model(y_pred)
        vgg_gt = vgg16_model(y_true)
        vgg_comp = vgg16_model(y_comp)

        #Compute loss components
        l1 = K.cast(loss_valid(x_mask, y_true, y_pred), 'float32')
        l2 = K.cast(loss_hole(x_mask, y_true, y_pred), 'float32')
        l3 = K.cast(loss_perceptual(vgg_out, vgg_gt, vgg_comp), 'float32')
        l4 = K.cast(loss_style(vgg_out, vgg_gt), 'float32')
        l5 = K.cast(loss_style(vgg_comp, vgg_gt), 'float32')
        l6 = K.cast(loss_tv(x_mask, y_comp), 'float32')

        # print("l1 = ", tf.math.reduce_mean(l1))
        # print("l2 = ", tf.math.reduce_mean(l2))
        # print("l3 = ", tf.math.reduce_mean(l3))  
        # print("l4 = ", tf.math.reduce_mean(l4))
        # print("l5 = ", tf.math.reduce_mean(l5))        
        # print("l6 = ", tf.math.reduce_mean(l6))

        # Return loss function
        lsum = tf.math.reduce_mean(l1 + 6*l2 + 0.05*l3 + 120*(l4+l5) + 0.1*l6)
        # lsum = tf.math.reduce_mean(l1 + 100*l2 + 0.05*l3 + 120*(l4+l5) + 0.1*l6)
        return lsum

    return loss

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = (*glob.glob("/home/ntustison/Data/CorticalThicknessData2014/*/T1/*.nii.gz"),
             *glob.glob("/home/ntustison/Data/SRPB1600/data/sub-*/t1/defaced_mprage.nii.gz"))

if len(t1_images) == 0:
    raise ValueError("NO training data.")
print("Total training image files: " + str(len(t1_images)))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 16

generator = batch_generator(batch_size=batch_size,
                            t1s=t1_images,
                            image_size=(image_size[0], image_size[1]),
                            number_of_channels=number_of_channels,
                            template=template,
                            template_labels=template_labels,
                            template_roi=template_roi,
                            template_priors=template_priors,
                            add_2d_masking=True,
                            do_histogram_intensity_warping=False,
                            do_simulate_bias_field=False,
                            do_add_noise=False,
                            do_data_augmentation=False
                            )

inpainting_weights_filename = scripts_directory + "t1_inpainting_with_priors_weights.h5"
if os.path.exists(inpainting_weights_filename):
    inpainting_unet.load_weights(inpainting_weights_filename)

inpainting_unet.compile()
# optimizer=keras.optimizers.SGD(learning_rate=0.02)
optimizer=keras.optimizers.Adam(learning_rate=2e-4)
train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()

number_of_epochs = 200
steps_per_epoch = 32
validation_steps = 1

minimum_value = 1000000000

for epoch in range(number_of_epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    for step in range(steps_per_epoch):
        x_batch_train, y_batch_train = next(generator)
        loss_fn=loss_total(x_batch_train[1][:,:,:,[0]])

        with tf.GradientTape() as tape:
            logits = inpainting_unet(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, inpainting_unet.trainable_weights)
        optimizer.apply_gradients(zip(grads, inpainting_unet.trainable_weights))
        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)
        # Log every 1 batches.
        if step % 1 == 0:
            print(
                "    Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("        Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    if float(train_acc) < minimum_value:
        print("Metric improved from " + str(minimum_value) + " to " + str(float(train_acc)))
        print("Saving " + inpainting_weights_filename)
        inpainting_unet.save_weights(inpainting_weights_filename)
        minimum_value = float(train_acc)

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    # Run a validation loop at the end of each epoch.
    for step in range(validation_steps):
        x_batch_val, y_batch_val = next(generator)
        val_logits = inpainting_unet(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))


