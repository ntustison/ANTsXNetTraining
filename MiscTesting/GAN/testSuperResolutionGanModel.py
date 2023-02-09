import requests
import tempfile
import numpy as np
import os
import cv2
import zipfile
import glob

import keras.backend as K

import ants
import antspynet

K.clear_session()

dataset = 'vangogh2photo'

data_directory = os.getcwd() + "/" + dataset
if not os.path.exists(data_directory):
    zipped_file = tempfile.NamedTemporaryFile(suffix=".zip", dir=os.getcwd())
    url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/" + dataset + ".zip"
    r = requests.get(url)
    with open(zipped_file.name, 'wb') as f:
        f.write(r.content)

    with zipfile.ZipFile(zipped_file.name, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())

scale_factor = 4
high_resolution_image_size = (256, 256, 3)
low_resolution_image_size = (int(high_resolution_image_size[0] / scale_factor),
                             int(high_resolution_image_size[1] / scale_factor),
                             3 )

gan_model = antspynet.SuperResolutionGanModel(
   low_resolution_image_size = low_resolution_image_size,
   scale_factor = scale_factor )

print("Reading training set A files.")
trainingA_files = glob.glob(data_directory + "/trainA/*.jpg")
number_of_trainingA_files = len(trainingA_files)
X_train_low_resolution = np.zeros(shape=(number_of_trainingA_files, *low_resolution_image_size))
X_train_high_resolution = np.zeros(shape=(number_of_trainingA_files, *high_resolution_image_size))

for i in range(number_of_trainingA_files):
   image = cv2.imread(trainingA_files[i])
   image = cv2.resize(image, high_resolution_image_size[:2])
   image = image / 127.5 - 1.0
   X_train_high_resolution[i,:,:,:] = image

   image = cv2.resize(image, low_resolution_image_size[:2])
   X_train_low_resolution[i,:,:,:] = image

print("Done.")

gan_model.train(X_train_low_resolution, X_train_high_resolution,
  number_of_epochs=200, batch_size=4, sample_interval=25,
  sample_file_prefix="./SuperResolutionGanSampleImages_Py/sample" )

