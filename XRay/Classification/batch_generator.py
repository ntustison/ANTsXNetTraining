import numpy as np
import random
import ants
import antspynet
import glob
import os

def batch_generator(batch_size,
                    image_files,
                    demo,
                    image_size=(224, 224),
                    population_prior=None,
                    do_augmentation=False):


    number_of_labels = demo.shape[1]
    number_of_channels = 3

    while True:

        X = np.zeros((batch_size, *image_size, number_of_channels))
        for i in range(batch_size):
            X[i,:,:,2] = population_prior.numpy()
        Y  = np.zeros((batch_size, demo.shape[1]))

        batch_count = 0
        while batch_count < batch_size:
            # randomly but uniformly choose a condition and then randomly select 
            # one of the images with that condition to ensure an equal sampling.
            random_condition = random.sample(list(range(number_of_labels)), 1)[0]
            condition_indices = np.where(demo[:, random_condition] == 1)[0]
            random_index = int(random.sample(list(condition_indices.flatten()), 1)[0])     
            base_image_file = image_files[random_index]
            base_image_file = base_image_file.replace(".png", ".nii.gz")
            image_file = glob.glob("/home/ntustison/Data/XRayCT/Data/Nifti/*/" + base_image_file)
            if len(image_file) > 0:
                image_file = image_file[0]
                mask_file = image_file.replace("Nifti", "Masks")
                if not os.path.exists(image_file) or not os.path.exists(mask_file):
                    continue
                image = ants.image_read(image_file)
                mask = ants.threshold_image(ants.image_read(mask_file), 0, 0, 0, 1)
                if image.components > 1:
                    image_channels = ants.split_channels(image)
                    image = (image_channels[0] + image_channels[1] + image_channels[2]) / 3
                image = ants.resample_image(image, image_size, use_voxels=True, interp_type=0)   
                if len(image.shape) == 2 and image.components == 1 and image.shape == image_size: 
                    mask = ants.resample_image(mask, image_size, use_voxels=True, interp_type=1)    
                    if do_augmentation:
                        sd_histogram_warping = 0.01
                        break_points = [0.2, 0.4, 0.6, 0.8]
                        displacements = list()
                        for b in range(len(break_points)):
                            displacements.append(random.gauss(0, sd_histogram_warping))
                        image = antspynet.histogram_warp_image_intensities(image,
                                            break_points=break_points,
                                            clamp_end_points=(True, True),
                                            displacements=displacements)               
                        data_aug = antspynet.randomly_transform_image_data(image,
                                                   [[image]], 
                                                   segmentation_image_list=[mask],
                                                   number_of_simulations=1,
                                                   transform_type="affineAndDeformation",
                                                   sd_affine=0.025,
                                                   deformation_transform_type="bspline",
                                                   number_of_random_points=50,
                                                   sd_noise=2.0,
                                                   mesh_size=4
                                                   )
                        image = data_aug['simulated_images'][0][0]
                        mask = data_aug['simulated_segmentation_images'][0]
                    image = (image - image.min()) / (image.max() - image.min())
                    # image = (image - image.mean()) / image.std()
                    X[batch_count,:,:,0] = image.numpy()
                    X[batch_count,:,:,1] = mask.numpy()
                    Y[batch_count,:] = demo[random_index,:]
                    batch_count += 1

        yield X, Y, None

