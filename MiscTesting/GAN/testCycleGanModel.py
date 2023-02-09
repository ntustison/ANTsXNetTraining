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

input_image_size = (128, 128, 3)

gan_model = antspynet.CycleGanModel(input_image_size=input_image_size)

print("Reading training set A files.")
trainingA_files = glob.glob(data_directory + "/trainA/*.jpg")
number_of_trainingA_files = len(trainingA_files)
X_trainA = np.zeros(shape=(number_of_trainingA_files, *input_image_size))

for i in range(number_of_trainingA_files):
   image = cv2.imread(trainingA_files[i])
   image = cv2.resize(image, input_image_size[:2])
   image = image / 127.5 - 1.0
   X_trainA[i,:,:,:] = image

print("Done.")


print("Reading training set B files.")
trainingB_files = glob.glob(data_directory + "/trainB/*.jpg")
number_of_trainingB_files = len(trainingB_files)
X_trainB = np.zeros(shape=(number_of_trainingB_files, *input_image_size))

for i in range(number_of_trainingB_files):
   image = cv2.imread(trainingB_files[i])
   image = cv2.resize(image, input_image_size[:2])
   image = image / 127.5 - 1.0
   X_trainB[i,:,:,:] = image

print("Done.")

gan_model.train(X_trainA, X_trainB, number_of_epochs=30000, sample_interval=100,
  batch_size=32, sample_file_prefix="./CycleGanSampleImages_Py/sample" )


