import ants
import numpy as np

from batch_generator import batch_generator

import os
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"

base_directory = '/home/ntustison/Data/DeepAtroposHCP/'
scripts_directory = base_directory + 'Scripts/'


data_directory = list()
data_directory.append(base_directory + "Data/HCP-A-3/")
data_directory.append(base_directory + "Data/HCP-YA-3/")

template_t1 = list()
template_t2 = list()
template_fa = list()
template_brain_mask = list()
template_segmentation = list()
template_priors = list()

for i in range(len(data_directory)):
    template_t1.append(ants.image_read(data_directory[i] + "T_template0.nii.gz"))
    template_t2.append(ants.image_read(data_directory[i] + "T_template1.nii.gz"))
    template_fa.append(ants.image_read(data_directory[i] + "T_template2.nii.gz"))
    template_brain_mask.append(ants.threshold_image(
                       ants.image_read(data_directory[i] + "T_templateBrainMask.nii.gz"),
                               0.5, 1.1, 1, 0))
    template_segmentation.append(ants.image_read(data_directory[i] + "T_templateBrainSegmentation_MT.nii.gz"))

    template_priors_local = list()
    for j in range(6):
        prior = ants.threshold_image(template_segmentation[i], j+1, j+1, 1, 0)
        prior_smooth = ants.smooth_image(prior, 1.0)
        template_priors_local.append(prior_smooth)
    template_priors.append(template_priors_local)    


###
#
# Set up the training generator
#
patch_size = (192, 224, 192)

batch_size = 8

generator = batch_generator(batch_size=batch_size,
                    template=template_t1,
                    priors=[],
                    template_modalities=[template_t1],
                    template_segmentation=template_segmentation,
                    template_brain_mask=template_brain_mask,
                    patch_size=patch_size,
                    number_of_octants_per_image=8,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    do_random_contralateral_flips=True,
                    do_resampling=True,
                    verbose=True)

X, Y = next(generator)

for i in range(X.shape[0]):
    print("Creating batch ", str(i))
    for j in range(X.shape[-1]):
        ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,j])), "batchX" + str(i) + "_" + str(j) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:])), "batchY_" + str(i) + ".nii.gz")

