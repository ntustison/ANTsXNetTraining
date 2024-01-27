import ants
import antspynet
import numpy as np


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ITK_DEFAULT_GLOBAL_NUMBER_OF_THREADS"] = "4"
import glob

base_directory = '/home/ntustison/Data/Lesions/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + 'ATLAS_2/'

from batch_whole_brain_generator import batch_generator

template_size = (192, 208, 192)
template = ants.image_read(antspynet.get_antsxnet_data('mni152'))
template = antspynet.pad_or_crop_image_to_size(template, template_size)
template_mask = antspynet.brain_extraction(template, modality="t1", verbose=True)

################################################
#
#  Load the brain data
#
################################################

print("Loading brain data.")

t1_files = glob.glob(data_directory + "Training/R*/sub*/ses*/anat/*T1w.nii.gz")

training_t1_files = list()
training_lesion_mask_files = list()
training_brain_mask_files = list()

for i in range(len(t1_files)):

    t1_file = t1_files[i] 
    lesion_mask_file = t1_file.replace("T1w.nii.gz", "label-L_desc-T1lesion_mask.nii.gz")
    brain_mask_file = t1_file.replace("T1w.nii.gz", "T1w_brain_mask.nii.gz")

    if (os.path.exists(t1_file) and 
        os.path.exists(lesion_mask_file) and
        os.path.exists(brain_mask_file)):

        training_t1_files.append(t1_file)
        training_lesion_mask_files.append(lesion_mask_file)
        training_brain_mask_files.append(brain_mask_file)
    else:
        print( "----> " + t1_file)    
        print( "      " + lesion_mask_file)    
        print( "      " + brain_mask_file)    

print("Total training image files: ", len(training_t1_files))

###
#
# Set up the training generator
#

batch_size = 8

generator = batch_generator(batch_size=batch_size,
                    template=template,
                    template_mask=template_mask,
                    t1_files=training_t1_files,
                    lesion_mask_files=training_lesion_mask_files,
                    brain_mask_files=training_brain_mask_files,                                  
                    do_histogram_intensity_warping=False,
                    do_simulate_bias_field=False,
                    do_add_noise=False,
                    do_random_transformation=False)

# patch_size = (64, 64, 64)
# generator = batch_generator(batch_size=batch_size,
#                     t1_files=training_t1_files,
#                     lesion_mask_files=training_lesion_mask_files,              
#                     brain_mask_files=training_brain_mask_files,
#                     patch_size=patch_size,
#                     do_histogram_intensity_warping=False,
#                     do_simulate_bias_field=False,
#                     do_add_noise=False,
#                     do_random_transformation=False)

X = next(generator)
print(X[0].shape)

for i in range(X[0].shape[0]):
    print("Creating batch ", str(i))
    ants.image_write(ants.from_numpy(np.squeeze(X[0][i,:,:,:,0])), "batchX_" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(X[1][i,:,:,:,0])), "batchY_" + str(i) + ".nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(Y[i,:,:,:,0])), "batchY_" + str(i) + ".nii.gz")    

print(X[0].shape)
# print(len(Y))


