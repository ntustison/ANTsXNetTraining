import ants
import antspynet
import numpy as np

import os
import glob

from batch_generator_test import batch_generator

base_directory = '/home/ntustison/Data/Cerebellum/'
scripts_directory = base_directory + 'Scripts/'
prior_directory = base_directory + "TemplateResampled0.5/"
data_directory = base_directory + "TrainingDataResampled0.5/"

priors = list()

# image_size = (233, 151, 154)
# image_size = (224, 128, 128)
image_size = (240, 144, 144)

# Load CSF/GM/WM labels
# for p in range(1, 4):
#     prior = ants.image_read(prior_directory + "T_template0Posteriors" + str(p) + ".nii.gz")
#     prior = antspynet.pad_or_crop_image_to_size(prior, image_size)
#     priors.append(prior)

# Load regional labels
prior_labels = (*list(range(1, 13)), *list(range(101, 113)))
simplified_prior_labels = (*list(range(1, 13)), *list(range(101, 113)))
for p in prior_labels:
    prior = ants.image_read(prior_directory + "T_template0_label_prior_" + str(p) + ".nii.gz")
    prior = antspynet.pad_or_crop_image_to_size(prior, image_size)
    priors.append(prior)

t1_template = ants.image_read(prior_directory + "T_template0N4.nii.gz")
t1_template = antspynet.pad_or_crop_image_to_size(t1_template, image_size)

################################################
#
#  Create the model and load weights
#
################################################

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = glob.glob(data_directory + "*region*.nii.gz")

training_t1_files = list()
training_labels_files = list()
training_tissue_files = list()

which_model = "whole"

for i in range(len(t1_images)):

    subject_directory = os.path.dirname(t1_images[i])
    if which_model == "labels_ala_dkt" or which_model == "labels_hybrid":
        labels_image = t1_images[i].replace("region", "simplified_labels")
    else:
        labels_image = t1_images[i].replace("region", "labels")

    tissue_image = t1_images[i].replace("region", "tissue")

    training_t1_files.append(t1_images[i])
    training_labels_files.append(labels_image)
    training_tissue_files.append(tissue_image)

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 10
if which_model == "labels_ala_dkt" or which_model == "labels_hybrid":
    labels = simplified_prior_labels
else:
    labels = prior_labels

generator = batch_generator(batch_size=batch_size,
                            t1s=training_t1_files,
                            t1_template=t1_template,
                            labels_images=training_labels_files,
                            tissue_images=training_tissue_files,
                            priors=priors,
                            regional_labels=labels,
                            do_histogram_intensity_warping=True,
                            do_simulate_bias_field=True,
                            do_add_noise=True,
                            do_data_augmentation=True,
                            which_model=which_model
                            )


X, Y, W = next(generator)

for i in range(X.shape[0]):
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,0]),
                     origin=t1_template.origin, spacing=t1_template.spacing,
                     direction=t1_template.direction), "batchX_" + str(i) + ".nii.gz")
    if which_model == "labels_ala_dkt" or which_model == "labels_hybrid":
        ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:]),
                     origin=t1_template.origin, spacing=t1_template.spacing,
                     direction=t1_template.direction), "batchY" + str(i) + ".nii.gz")
    elif which_model == "whole":
        ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:]),
                     origin=t1_template.origin, spacing=t1_template.spacing,
                     direction=t1_template.direction), "batchY2" + str(i) + ".nii.gz")
    elif which_model == "tissue":    
        ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:]),
                     origin=t1_template.origin, spacing=t1_template.spacing,
                     direction=t1_template.direction), "batchY3" + str(i) + ".nii.gz")
    else:
        ants.image_write(ants.from_numpy(np.squeeze(Y[0][i,:,:,:]),
                     origin=t1_template.origin, spacing=t1_template.spacing,
                     direction=t1_template.direction), "batchY" + str(i) + ".nii.gz")
        ants.image_write(ants.from_numpy(np.squeeze(Y[1][i,:,:,:]),
                     origin=t1_template.origin, spacing=t1_template.spacing,
                     direction=t1_template.direction), "batchY2" + str(i) + ".nii.gz")
        ants.image_write(ants.from_numpy(np.squeeze(Y[2][i,:,:,:]),
                     origin=t1_template.origin, spacing=t1_template.spacing,
                     direction=t1_template.direction), "batchY3" + str(i) + ".nii.gz")


print(X.shape)
print(len(Y))


