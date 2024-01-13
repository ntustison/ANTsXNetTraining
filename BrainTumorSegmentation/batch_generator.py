import numpy as np
import random
import ants
import antspynet
import os

import time

def batch_generator(batch_size=32,
                    flair_files=None,
                    t1_files=None,
                    t1gd_files=None,
                    t2_files=None,
                    brain_mask_files=None,
                    seg_files=None,
                    patch_size=(64, 64, 64),
                    number_of_channels=4,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=False):


    number_of_patches_per_image = 4

    while True:

        X = np.zeros((batch_size, *patch_size, number_of_channels))
        Y = np.zeros((batch_size, *patch_size))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(t1_files))), 1)[0]

            t1 = ants.image_read(t1_files[i])
            t2 = ants.image_read(t2_files[i])
            t1gd = ants.image_read(t1gd_files[i])
            flair = ants.image_read(flair_files[i])

            images = [flair, t1, t1gd, t2]

            mask = ants.image_read(brain_mask_files[i])
            seg = ants.image_read(seg_files[i])
            
            if do_histogram_intensity_warping: # and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                break_points = [0.2, 0.4, 0.6, 0.8]                
                for ii in range(len(images)):
                    displacements = list()
                    for b in range(len(break_points)):
                        displacements.append(abs(random.gauss(0, 0.05)))
                        if random.sample((True, False), 1)[0]:
                            displacements[b] *= -1
                    images[ii] = antspynet.histogram_warp_image_intensities(images[ii],
                        break_points=break_points, clamp_end_points=(True, False),
                        displacements=displacements)
                toc = time.perf_counter()
                # print(f"Histogram warping {toc - tic:0.4f} seconds")

            if do_add_noise and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                for ii in range(len(images)):
                    images[ii] = (images[ii] - images[ii].min()) / (images[ii].max() - images[ii].min())
                    noise_parameters = (0.0, random.uniform(0, 0.01))
                    images[ii] = ants.add_noise_to_image(images[ii], noise_model="additivegaussian", noise_parameters=noise_parameters)
                toc = time.perf_counter()
                # print(f"Add noise {toc - tic:0.4f} seconds")

            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                for ii in range(len(images)):
                    log_field = antspynet.simulate_bias_field(images[ii], number_of_points=10, sd_bias_field=1.0, number_of_fitting_levels=2, mesh_size=10)
                    log_field = log_field.iMath("Normalize")
                    field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3, 4), 1)[0])
                    images[ii] = images[ii] * ants.from_numpy(field_array, origin=images[ii].origin, spacing=images[ii].spacing, direction=images[ii].direction)
                    images[ii] = (images[ii] - images[ii].min()) / (images[ii].max() - images[ii].min())
                toc = time.perf_counter()
                # print(f"Sim bias field {toc - tic:0.4f} seconds")

            if do_random_transformation:
                tic = time.perf_counter()
                data_augmentation = antspynet.randomly_transform_image_data(t1,
                    [images],
                    [seg],
                    number_of_simulations=1,
                    transform_type='deformation',
                    sd_affine=0.01,
                    deformation_transform_type="bspline",
                    number_of_random_points=1000,
                    sd_noise=5.0,
                    number_of_fitting_levels=4,
                    mesh_size=1,
                    sd_smoothing=4.0,
                    input_image_interpolator='linear',
                    segmentation_image_interpolator='nearestNeighbor')
                mask = ants.apply_ants_transform_to_image(data_augmentation['simulated_transforms'][0], mask,
                    reference=t1, interpolation='nearestneighbor')
                images = data_augmentation['simulated_images'][0]
                seg = data_augmentation['simulated_segmentation_images'][0]
                toc = time.perf_counter()
                # print(f"Xfrms {toc - tic:0.4f} seconds")

            tic = time.perf_counter()
            for ii in range(len(images)):
                images[ii] = images[ii] * mask
                indices = mask > 0
                images[ii] = (images[ii] - images[ii][indices].min()) / (images[ii][indices].max() - images[ii][indices].min())
            toc = time.perf_counter()
            # print(f"Misc {toc - tic:0.4f} seconds")

            tic = time.perf_counter()
            random_seed = random.randint(0, 1e8)
            
            seg_mask = ants.threshold_image(seg, 0, 0, 0, 1)
            if seg_mask.sum() < 1:
                continue

            image_patches = list()
            for ii in range(len(images)):
                image_patches.append(antspynet.extract_image_patches(images[ii], 
                                                         patch_size=patch_size,
                                                         max_number_of_patches=number_of_patches_per_image,
                                                         mask_image=seg_mask,
                                                         random_seed=random_seed,
                                                         return_as_array=True))
            seg_patches = antspynet.extract_image_patches(seg, 
                                                          patch_size=patch_size,
                                                          max_number_of_patches=number_of_patches_per_image,
                                                          mask_image=seg_mask,
                                                          random_seed=random_seed,
                                                          return_as_array=True)

            for p in range(number_of_patches_per_image):
                seg_patch = np.squeeze(seg_patches[p,:,:,:]) 

                image_patch = list()            
                for ii in range(len(images)):
                    image_patch.append(np.squeeze(image_patches[ii][p,:,:,:]))
                                             
                number_of_swaps = random.randint(0, 3)
                for _ in range(number_of_swaps):
                    axes = random.sample((0, 1, 2), 2)
                    for ii in range(len(images)):
                        image_patch[ii] = np.swapaxes(image_patch[ii], axes[0], axes[1])
                    seg_patch = np.swapaxes(seg_patch, axes[0], axes[1])
                for axis in range(3):
                    if random.sample((True, False), 1)[0]:
                        for ii in range(len(images)):
                            image_patch[ii] = np.flip(image_patch[ii], axis)
                        seg_patch = np.flip(seg_patch, axis) 
                toc = time.perf_counter()
                # print(f"Patches {toc - tic:0.4f} seconds")

                for ii in range(len(images)): 
                    X[batch_count,:,:,:,ii] = image_patch[ii]
                seg_patch_one_label = np.zeros(seg_patch.shape)
                seg_patch_one_label[seg_patch > 0.5] = 1.0
                X[batch_count,:,:,:,number_of_channels-1] = seg_patch_one_label    
                Y[batch_count,:,:,:] = seg_patch
                
                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break
             
            segmentation_labels = list(range(number_of_channels)) 
            Y_enc = antspynet.encode_unet(Y, segmentation_labels) 

        #yield X, Y, None
        yield X, Y_enc, None






