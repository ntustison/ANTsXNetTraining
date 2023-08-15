import ants
import antspynet
import numpy as np
import random
import glob
import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

base_directory = '/Users/ntustison/Data/Public/XRayCT/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + "Data/"

################################################
#
#  Load the data
#
################################################

test_images_file = base_directory + "CXR8-selected/test_list.txt"
with open(test_images_file) as f:
    test_images_list = f.readlines()
f.close()
test_images_list = [x.strip() for x in test_images_list]

demo_file = base_directory + "CXR8-selected/Data_Entry_2017_v2020.csv"
demo = pd.read_csv(demo_file)

def unique(list1):
    unique_list = [] 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return sorted(unique_list)

unique_labels = demo['Finding Labels'].unique()
unique_labels_unroll = []
for i in range(len(unique_labels)):
    label = unique_labels[i]
    labels = label.split('|')
    for j in range(len(labels)):
        unique_labels_unroll.append(labels[j])

unique_labels = unique(unique_labels_unroll)

testing_demo_file = base_directory + "testing_demo.npy"
testing_demo = None
if os.path.exists(testing_demo_file):
    testing_demo = np.load(testing_demo_file)
else:
    testing_demo = np.zeros((len(test_images_list), len(unique_labels)))    
    for i in tqdm(range(len(test_images_list))):
        image_filename = test_images_list[i]
        row = demo.loc[demo['Image Index'] == image_filename]
        findings = row['Finding Labels'].str.cat().split("|")
        for j in range(len(findings)):
            testing_demo[i, unique_labels.index(findings[j])] = 1.0            
    np.save(testing_demo_file, testing_demo)    

################################################
#
#  Create the model and load weights
#
################################################

image_size=(1024, 1024)

number_of_classification_labels = len(unique_labels)

model = antspynet.create_resnet_model_2d((None, None, 1),
   number_of_classification_labels=number_of_classification_labels,
   mode="regression",
   layers=(1, 2, 3, 4),
   residual_block_schedule=(2, 2, 2, 2), lowest_resolution=64,
   cardinality=1, squeeze_and_excite=False)

weights_filename = scripts_directory + "xray_classification_with_augmentation.h5"
model.load_weights(weights_filename)

################################################
#
#  Test the model
#
################################################

batchX = np.zeros((1,*image_size,1))    

for i in range(10):
    random_index = random.sample(list(range(len(test_images_list))), 1)[0]
    image_file = glob.glob(data_directory + "/*/" + test_images_list[random_index])
    if len(image_file) > 0:
        image_file = image_file[0]
        image = ants.image_read(image_file)
        if len(image.shape) == 2 and image.components == 1:       
            print(image_file)
            image = ants.resample_image(image, image_size, use_voxels=True)        
            image = (image - image.min()) / (image.max() - image.min())
            batchX[0,:,:,0] = image.numpy()            
            Y = (tf.nn.sigmoid(model.predict(batchX, verbose=False))).numpy()
            Y = np.squeeze(Y)
            index = np.argmax(Y)
            Y[:] = 0
            Y[index] = 1.
            print("  actual: ", testing_demo[random_index,:])
            print("  predic: ", Y)
            
