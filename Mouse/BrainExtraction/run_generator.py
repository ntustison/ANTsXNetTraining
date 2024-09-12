import ants
import antspynet
import numpy as np

import os
import glob

from batch_generator import batch_generator

base_directory = '/home/ntustison/Data/Mouse/BrainExtraction/3D/'

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

template1_file = base_directory + "bspline_template.nii.gz"
mask1_file = base_directory + "bspline_template_mask.nii.gz"

template1 = ants.resample_image(ants.image_read(template1_file), (176, 176, 176), use_voxels=True, interp_type=0)
mask1 = ants.resample_image(ants.image_read(mask1_file), (176, 176, 176), use_voxels=True, interp_type=1)

ants.set_spacing(template1, (1, 1, 1))
ants.set_spacing(mask1, (1, 1, 1))

image_size = template1.shape

template1 = ants.iMath_normalize(template1)

### Template 2

template2_file = base_directory + "HR_template.nii.gz"
mask2_file = base_directory + "HR_template_mask.nii.gz"

template2 = ants.resample_image(ants.image_read(template2_file), (176, 176, 176), use_voxels=True, interp_type=0)
mask2 = ants.resample_image(ants.image_read(mask2_file), (176, 176, 176), use_voxels=True, interp_type=1)

ants.set_spacing(template2, (1, 1, 1))
ants.set_spacing(mask2, (1, 1, 1))

template2 = ants.iMath_normalize(template2)

### Template 3

template3_file = base_directory + "araikes_template.nii.gz"
mask3_file = base_directory + "araikes_template_mask.nii.gz"

template3 = ants.resample_image(ants.image_read(template3_file), (176, 176, 176), use_voxels=True, interp_type=0)
mask3 = ants.resample_image(ants.image_read(mask3_file), (176, 176, 176), use_voxels=True, interp_type=1)

ants.set_spacing(template3, (1, 1, 1))
ants.set_spacing(mask3, (1, 1, 1))

template3 = ants.iMath_normalize(template3)

###
# 
# Set up the training generator
#

batch_size = 10

generator = batch_generator(batch_size=batch_size,
                    template=template1,
                    images=[template1, template2, template3],
                    labels=[mask1, mask2, mask3],
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    resample_direction="random")

X, Y, W = next(generator)


template_shape = (176, 176, 176) 
template = ants.image_read(antspynet.get_antsxnet_data("bsplineT2MouseTemplate"))
template = ants.resample_image(template, template_shape, use_voxels=True, interp_type=0)

for i in range(X.shape[0]):
    print("Creating batch ", str(i))
    batchX = ants.from_numpy_like(np.squeeze(X[i,:,:,:,0]), template)
    batchY = ants.from_numpy_like(np.squeeze(Y[i,:,:,:]), template)
    batchP = antspynet.mouse_brain_extraction(batchX, modality="t2", verbose=True)
    ants.image_write(batchX, "batchX_" + str(i) + ".nii.gz")
    ants.image_write(batchY, "batchY_" + str(i) + ".nii.gz")
    ants.image_write(batchP, "batchP_" + str(i) + ".nii.gz")

# print(X.shape)
# print(len(Y))


