import ants
import antspynet
import numpy as np
import glob

import os
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "2"

from batch_combined_generator import batch_generator


base_directory = '/home/ntustison/Data/HoaLabeling/'
scripts_directory = base_directory + 'Scripts/'
priors_directory = base_directory + "Data/PriorProbabilityImages/"
labels_directory = base_directory + "Data/SubcorticalParcellations/dseg2/"
crop_size = (160, 176, 160)

labels = None
batch_size = None
batch_size = 4  

template_t1_files = list()
template_segmentation_files = list()

def reshape_image(image, interp_type = "linear", crop_size=crop_size):
    image_resampled = None
    if interp_type == "linear":
        image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=0)
    else:        
        image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=1)
    image_cropped = antspynet.pad_or_crop_image_to_size(image_resampled, crop_size)
    return image_cropped

template = reshape_image(ants.image_read(priors_directory + "prior1.nii.gz"))

t1_files = glob.glob(base_directory + "Data/T1w*/*T1w_extracted.nii.gz")

for i in range(len(t1_files)):
    t1_file = t1_files[i]
    labels_file = t1_file.replace("T1w_extracted.nii.gz", "dseg2.nii.gz")
    template_t1_files.append(t1_file)
    template_segmentation_files.append(labels_file)

print("Training data size: " + str(len(template_t1_files)))


###
#
# Set up the training generator
#

generator = batch_generator(batch_size=batch_size,
                    template=template,
                    crop_size=crop_size,
                    prior_files=[],
                    training_image_files=template_t1_files,
                    training_segmentation_files=template_segmentation_files,
                    classification_labels=labels,
                    do_histogram_intensity_warping=True,
                    do_histogram_equalization=False,
                    do_histogram_rank_intensity=False,
                    do_simulate_bias_field=True,
                    do_add_noise=False,
                    do_random_transformation=True,
                    do_random_contralateral_flips=True,
                    do_resampling=True,
                    use_multiple_outputs=True,
                    verbose=True)

X, Y = next(generator)

for i in range(X.shape[0]):
    print("Creating batch ", str(i))
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,0])), "batchX" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:])), "batchY_" + str(i) + ".nii.gz")

