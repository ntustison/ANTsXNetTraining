import ants
import antspynet
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import glob

from batch_generator import batch_generator

base_directory = '/Users/ntustison/Data/Public/XRayCT/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + "Data/"

################################################
#
#  Load the data
#
################################################

train_images_file = base_directory + "CXR8-selected/train_val_list.txt"
with open(train_images_file) as f:
    train_images_list = f.readlines()
f.close()
train_images_list = [x.strip() for x in train_images_list]

demo2017_file = base_directory + "CXR8-selected/BBox_List_2017.csv"
demo2017 = pd.read_csv(demo2017_file)

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

training_demo_file = base_directory + "training_demo.npy"
training_demo = None
if os.path.exists(training_demo_file):
    training_demo = np.load(training_demo_file)
else:
    training_demo = np.zeros((len(train_images_list), len(unique_labels)))
    for i in tqdm(range(len(train_images_list))):
        image_filename = train_images_list[i]
        row = demo.loc[demo['Image Index'] == image_filename]
        findings = row['Finding Labels'].str.cat().split("|")
        for j in range(len(findings)):
            training_demo[i, unique_labels.index(findings[j])] = 1.0
    np.save(training_demo_file, training_demo)        
##
#
# Set up the training generator
#

batch_size = 10

generator = batch_generator(batch_size=batch_size,
                            image_files=train_images_list,
                            demo=training_demo,
                            do_augmentation=True)


X, Xpre, Y, W = next(generator)

for i in range(X.shape[0]):
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,0]), origin=(0, 0),
                                     spacing=(1, 1), direction=np.eye(2)), "batch" + str(i) + "_X.nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(Xpre[i,:,:,0]), origin=(0, 0),
                                     spacing=(1, 1), direction=np.eye(2)), "batch" + str(i) + "_Xpre.nii.gz")

print(X.shape)
print(len(Y))


