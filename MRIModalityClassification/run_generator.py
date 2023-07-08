import ants
import antspynet
import numpy as np

import glob

from batch_generator import batch_generator

base_directory = '/Users/ntustison/Pkg/ANTsXNetTraining/MRIModalityClassification/'
scripts_directory = base_directory
data_directory = '/Users/ntustison/Data/Public/OpenNeuro/NIMHHealthyVolunteer/'

# image_size = (224, 224, 224)
# resample_size = (1, 1, 1)

image_size = (112, 112, 112)
resample_size = (2, 2, 2)

template = ants.image_read(antspynet.get_antsxnet_data("kirby"))
template = ants.resample_image(template, resample_size)
template = antspynet.pad_or_crop_image_to_size(template, image_size)
direction = template.direction
direction[0, 0] = 1.0
ants.set_direction(template, direction)
ants.set_origin(template, (0, 0, 0))

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = glob.glob(data_directory + "**/*T1w.nii.gz", recursive=True)
t2_images = glob.glob(data_directory + "**/*T2w.nii.gz", recursive=True)
flair_images = glob.glob(data_directory + "**/*FLAIR.nii.gz", recursive=True)
t2star_images = glob.glob(data_directory + "**/*T2starw.nii.gz", recursive=True)
dwi_images = glob.glob(data_directory + "**/*MeanDwi.nii.gz", recursive=True)
bold_images = glob.glob(data_directory + "**/*MeanBold.nii.gz", recursive=True)
perf_images = glob.glob(data_directory + "**/*asl.nii.gz", recursive=True)

images = t1_images + t2_images + flair_images + t2star_images + dwi_images + bold_images + perf_images
modalities = np.concatenate((
             np.zeros((len(t1_images),), dtype=np.int8),
             np.zeros((len(t2_images),), dtype=np.int8) + 1,
             np.zeros((len(flair_images),), dtype=np.int8) + 2,
             np.zeros((len(t2star_images),), dtype=np.int8) + 3,
             np.zeros((len(dwi_images),), dtype=np.int8) + 4,
             np.zeros((len(bold_images),), dtype=np.int8) + 5,
             np.zeros((len(perf_images),), dtype=np.int8) + 6),
             dtype=np.int8
             )

print( "Training")

###
#
# Set up the training generator
#

batch_size = 10

generator = batch_generator(batch_size=batch_size,
                            image_files=images,
                            template=template,
                            modalities=modalities
                            )


X, Y, W = next(generator)

for i in range(X.shape[0]):
    print("Writing image ", str(i))
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,0]),
                     origin=template.origin, spacing=template.spacing,
                     direction=template.direction), "batchX_" + str(i) + ".nii.gz")

