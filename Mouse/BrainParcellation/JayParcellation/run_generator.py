import ants
import antspynet
import numpy as np

from batch_generator import batch_generator

base_directory = './'

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

template_file = base_directory + "DevCCF_P04_STPT_50um.nii.gz"
labels_file = base_directory + "DevCCF_P04_STPT_50um_BrainParcellationJayMask.nii.gz"

template = ants.image_read(template_file)
labels = ants.image_read(labels_file)

template = ants.resample_image(template, (75, 75, 75), use_voxels=False, interp_type=4)
labels = ants.resample_image(labels, (75, 75, 75), use_voxels=False, interp_type=1)

template = antspynet.pad_or_crop_image_to_size(template, (176, 128, 240))
labels = antspynet.pad_or_crop_image_to_size(labels, (176, 128, 240))

ants.set_spacing(template, (1, 1, 1))
ants.set_spacing(labels, (1, 1, 1))

image_size = template.shape
unique_labels = np.unique(labels.numpy())
number_of_classification_labels = len(unique_labels)
number_of_nonzero_labels = number_of_classification_labels - 1
channel_size = 1 + number_of_nonzero_labels

print("Unique labels: ", unique_labels)

template_priors = list()
for i in range(number_of_nonzero_labels):
    single_label = ants.threshold_image(labels, i+1, i+1)
    prior = ants.smooth_image(single_label, sigma=3, sigma_in_physical_coordinates=True)
    template_priors.append(prior)

template = (template - template.min()) / (template.max() - template.min()) 

###
#
# Set up the training generator
#

batch_size = 10

generator = batch_generator(batch_size=batch_size,
                    template=template,
                    template_priors=template_priors,
                    images=[template],
                    labels=[labels],
                    unique_labels=unique_labels,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    resample_direction="random")

X, Y, W = next(generator)

for i in range(X.shape[0]):
    print("Creating batch ", str(i))
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,0])), "batchX_" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:])), "batchY_" + str(i) + ".nii.gz")

print(X.shape)
print(len(Y))



