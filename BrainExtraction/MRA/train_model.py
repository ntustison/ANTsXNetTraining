import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"
import glob

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_generator import batch_generator

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()
base_directory = '/home/ntustison/Data/MRA/BrainExtraction/'
scripts_directory = base_directory

from batch_generator import batch_generator

template = ants.image_read(antspynet.get_antsxnet_data("S_template3"))
template_brain_mask = antspynet.brain_extraction(template, modality="t1")

template_size = template.shape

################################################
#
#  Create the model and load weights
#
################################################

classes = ['background', 'brain']
number_of_classification_labels = len(classes)
image_modalities = ["T1"]
channel_size = len(image_modalities)

binary_dice_loss = antspynet.binary_dice_coefficient(smoothing_factor=0.0)
surface_loss = antspynet.binary_surface_loss()
ce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

unet_model = antspynet.create_unet_model_3d((*template_size, channel_size),
    mode="sigmoid",
    number_of_outputs=1,
    number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
    convolution_kernel_size=3, deconvolution_kernel_size=2,
    weight_decay=1e-5)

weights_filename = scripts_directory + "brainExtractionMra.h5"
if os.path.exists(weights_filename):
    unet_model.load_weights(weights_filename)

unet_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                   loss=binary_dice_loss,
                   metrics=[binary_dice_loss])


################################################
#
#  Load the brain data
#
################################################

print("Loading braindata.")

base_data_directory = base_directory + '/../ITKTubeTK/TemplateNifti/'
mask_images = glob.glob(base_data_directory + "Normal*/BrainMask.nii.gz")

training_image_files = list()
training_mask_files = list()

for i in range(len(mask_images)):
    mask = mask_images[i]
    image = mask.replace("BrainMask", "MRA")

    if not os.path.exists(image) or not os.path.exists(mask):
        # print(mask + " ---> " + image)
        continue

    training_image_files.append(image)
    training_mask_files.append(mask)

base_data_directory = base_directory + '../brains/'
mask_images = glob.glob(base_data_directory + "Masks2/*.nii.gz")

for i in range(len(mask_images)):
    mask = mask_images[i]
    image = mask.replace("Masks2", "Nifti")

    if not os.path.exists(image) or not os.path.exists(mask):
        continue

    training_image_files.append(image)
    training_mask_files.append(mask)


print("Total training image files: ", len(training_image_files))
print( "Training")


###
#
# Set up the training generator
#

batch_size = 4


generator = batch_generator(batch_size=batch_size,
                            image_size=template_size,
                            template=template,
                            template_brain_mask=template_brain_mask,                            
                            images=training_image_files,
                            brain_masks=training_mask_files,
                            do_random_contralateral_flips=True,
                            do_histogram_intensity_warping=True,
                            do_simulate_bias_field=True,
                            do_add_noise=True,
                            do_data_augmentation=True)


track = unet_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95,
          verbose=1, patience=20, mode='auto'),
    #    keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001,
    #       patience=20)
       ]
   )

unet_model.save_weights(weights_filename)



