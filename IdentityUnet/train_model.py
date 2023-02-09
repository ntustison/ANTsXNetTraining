import ants
import antspynet

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import glob
import numpy as np
import time

from batch_generator import batch_generator

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

base_directory = '/home/ntustison/Data/IdentityUnet/'
scripts_directory = base_directory

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

use_partial_conv = True
unet_model = antspynet.create_partial_convolution_unet_model_2d((*image_size, channel_size),
                                                                number_of_priors=0,
                                                                number_of_filters=(16, 32, 64, 128),
                                                                kernel_size=3,
                                                                use_partial_conv=use_partial_conv)


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


if use_partial_conv:
    weights_filename = scripts_directory + "t1_pconv_identity_unet_weights.h5"
else:
    weights_filename = scripts_directory + "t1_no_pconv_identity_unet_weights.h5"

if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                  loss="mse")

optimizer=keras.optimizers.Adam(learning_rate=2e-4)
train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()

###
#
# Set up the training generator
#

batch_size = 16

generator = batch_generator(batch_size=batch_size,
                            t1s=t1_images,
                            image_size=(image_size[0], image_size[1]),
                            number_of_channels=channel_size,
                            template=template,
                            do_histogram_intensity_warping=False,
                            do_simulate_bias_field=False,
                            do_add_noise=False,
                            do_data_augmentation=False,
                            return_one_masks=True
                            )

number_of_epochs = 200
steps_per_epoch = 32
validation_steps = 1

minimum_value = 1000000000

def loss():
    def mse_loss(y_pred, y_true):
        return tf.reduce_mean(tf.square(y_pred - y_true))
    return mse_loss

loss_fn = loss()

for epoch in range(number_of_epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    for step in range(steps_per_epoch):
        x_batch_train, y_batch_train = next(generator)

        with tf.GradientTape() as tape:
            logits = unet_model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, unet_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, unet_model.trainable_weights))
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
        print("Saving " + weights_filename)
        unet_model.save_weights(weights_filename)
        minimum_value = float(train_acc)

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
    # Run a validation loop at the end of each epoch.
    for step in range(validation_steps):
        x_batch_val, y_batch_val = next(generator)
        val_logits = unet_model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))


# track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
#     callbacks=[
#        keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
#            save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
#        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
#           verbose=1, patience=20, mode='auto'),
#        keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001,
#           patience=40)
#        ]
#   )
