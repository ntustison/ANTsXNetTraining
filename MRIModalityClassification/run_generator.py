import ants
import antspynet
import numpy as np

import glob

from batch_generator import batch_generator

base_directory = '/Users/ntustison/Data/'
data_directory = base_directory + "CorticalThicknessData2014/Kirby/BIDS/"

t1_template = ants.image_read(antspynet.get_antsxnet_data("kirby"))

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

image_files = glob.glob(data_directory + "sub*/anat/sub*T1w.nii.gz")
modalities = np.zeros((len(image_files),)) + 1
modalities = modalities.astype(int)

print("Total training image files: ", len(image_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 10

generator = batch_generator(batch_size=batch_size,
                            image_files=image_files,
                            template=t1_template,
                            modalities=modalities
                            )


X, Y, W = next(generator)

for i in range(X.shape[0]):
    ants.image_write(ants.from_numpy(np.squeeze(X[i,:,:,:,0]),
                     origin=t1_template.origin, spacing=t1_template.spacing,
                     direction=t1_template.direction), "batchX_" + str(i) + ".nii.gz")

