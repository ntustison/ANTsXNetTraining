import numpy as np
import random
import ants
import antspynet
import glob
import os

def batch_generator(batch_size,
                    demo,
                    do_augmentation=False):

    # use imagenet mean,std for normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    image_size = (224, 224)
    number_of_channels = 3

    disease_categories = ['Atelectasis',
                          'Cardiomegaly',
                          'Effusion',
                          'Infiltration',
                          'Mass',
                          'Nodule',
                          'Pneumonia',
                          'Pneumothorax',
                          'Consolidation',
                          'Edema',
                          'Emphysema',
                          'Fibrosis',
                          'Pleural_Thickening',
                          'Hernia']
    number_of_dx = len(disease_categories)

    while True:

        X = np.zeros((batch_size, *image_size, number_of_channels))
        Y  = np.zeros((batch_size, number_of_dx))
        
        batch_count = 0
        while batch_count < batch_size:
            random_index = int(random.sample(list(range(demo.shape[0])), 1)[0])
            subject_row = demo.iloc[[random_index]]
            base_image_file = subject_row.index.values[0]
            base_image_file = base_image_file.replace(".png", ".nii.gz")
            image_file = glob.glob("/home/ntustison/Data/reproduce-chexnet/data/nifti/" + base_image_file)
            if len(image_file) > 0:
                image_file = image_file[0]
                mask_file = image_file.replace("nifti", "masks")
                if not os.path.exists(image_file) or not os.path.exists(mask_file):
                    continue
                image = ants.image_read(image_file)
                mask = ants.image_read(mask_file)
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
                    image_array = image.numpy()
                    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
                    X[batch_count,:,:,0] = (image_array - imagenet_mean[0]) / (imagenet_std[0])
                    X[batch_count,:,:,1] = (image_array - imagenet_mean[1]) / (imagenet_std[1])
                    X[batch_count,:,:,1] *= (ants.threshold_image(mask, 1, 1, 1, 0)).numpy() 
                    X[batch_count,:,:,2] = (image_array - imagenet_mean[2]) / (imagenet_std[2])
                    X[batch_count,:,:,2] *= (ants.threshold_image(mask, 2, 2, 1, 0)).numpy() 
                    for d in range(number_of_dx):    
                        Y[batch_count,d] = subject_row[disease_categories[d]].values[0]
                    batch_count += 1

        yield X, Y, None

