import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from batch_vgg16_generator import batch_generator

K.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

base_directory = '/home/ntustison/Data/CorticalThicknessData2014/'
scripts_directory = base_directory + 'Training/'

template = ants.image_read(antspynet.get_antsxnet_data("oasis"))

################################################
#
#  Create the model and load weights
#
################################################

image_modalities = ["T1", "T1", "T1"]
channel_size = 3
template_size = template.shape

image_size=(224, 224)

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
                                             mode='regression')
vgg16_model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                   loss="mse")

weights_filename = scripts_directory + "vgg16_imagenet_t1brain_weights.h5"
if os.path.exists(weights_filename):
    vgg16_model.load_weights(weights_filename)
else:
    vgg16_keras = keras.applications.VGG16(include_top=True,
                                       weights='imagenet',
                                       input_tensor=None,
                                       input_shape=None,
                                       pooling=None,
                                       classes=1000,
                                       classifier_activation='softmax')
    for i in range(len(vgg16_keras.layers)-1):
        vgg16_model.layers[i].set_weights(vgg16_keras.layers[i].get_weights())
    

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_files = glob.glob(base_directory + "*/T1/*.nii.gz")

t1_images = list()
t1_ages = list()

for i in range(len(t1_files)):
    csv_file = t1_files[i].replace(".nii.gz", ".csv")
    if os.path.exists(csv_file):
        try:
            print(csv_file)
            t1_pd = pd.read_csv(csv_file)
            if t1_pd.shape[0] > 0:
                if 'Age' in t1_pd:
                    value = t1_pd['Age'].iloc[0]
                if 'AGE' in t1_pd:
                    value = t1_pd['AGE'].iloc[0]
                if np.isfinite(value):
                    t1_images.append(t1_files[i])
                    t1_ages.append(int(value))
        except TypeError:
            pass

# Also load the SRPB data

t1_files = glob.glob("/home/ntustison/Data/SRPB1600/data/sub-*/t1/defaced_mprage.nii.gz")
demo = pd.read_csv("/home/ntustison/Data/SRPB1600/participants_diagscore/participants_decnef-Table 1.csv")

for i in range(len(t1_files)):
    print(t1_files[i])
    subject_directory = os.path.dirname(t1_files[i])
    subject_id = subject_directory.split('/')[6]    
    subject_row = demo.loc[demo['participants_id'] == subject_id]
    if not subject_row.empty:
        value = subject_row['age'].iloc[0]  
        if np.isfinite(value):
            t1_images.append(t1_files[i])
            t1_ages.append(int(value))

if len(t1_images) == 0:
    raise ValueError("NO training data.")
print("Total training image files: ", len(t1_images))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 32

generator = batch_generator(batch_size=batch_size,
                            t1s=t1_images,
                            t1_ages=t1_ages,
                            image_size=image_size,
                            template=template,
                            do_histogram_intensity_warping=False,
                            do_simulate_bias_field=False,
                            do_add_noise=False,
                            do_data_augmentation=False
                            )


track = vgg16_model.fit(x=generator, epochs=200, verbose=1, steps_per_epoch=32,
    callbacks=[
       keras.callbacks.ModelCheckpoint(weights_filename, monitor='loss',
           save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
       keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
          verbose=1, patience=10, mode='auto'),
       keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001,
          patience=20)
       ]
   )

