import numpy as np
import random
import ants
import antspynet
import os

import time

def batch_generator(batch_size=32,
                    ct_files=None,
                    seg_files=None,
                    label_files=None,
                    patch_size=(64, 64, 64),
                    number_of_channels=4,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=False):


    number_of_patches_per_image = 4

    while True:

        X = np.zeros((batch_size, *patch_size, number_of_channels))
        Y = np.zeros((batch_size, *patch_size, 1))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(ct_files))), 1)[0]

            ct = ants.image_read(ct_files[i])
            images = list()
            images.append(ct)
            seg = ants.image_read(seg_files[i])
            labels = ants.image_read(label_files[i])

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
                data_augmentation = antspynet.randomly_transform_image_data(ct,
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
                labels = ants.apply_ants_transform_to_image(data_augmentation['simulated_transforms'][0], labels,
                    reference=ct, interpolation='nearestneighbor')
                images = data_augmentation['simulated_images'][0]
                seg = data_augmentation['simulated_segmentation_images'][0]
                toc = time.perf_counter()
                # print(f"Xfrms {toc - tic:0.4f} seconds")

            tic = time.perf_counter()
            for ii in range(len(images)):
                images[ii] = (images[ii] + 800) / (500 + 800)
                images[ii][images[ii] > 1.0] = 1.0
                images[ii][images[ii] < 0.0] = 0.0
            toc = time.perf_counter()
            # print(f"Misc {toc - tic:0.4f} seconds")

            tic = time.perf_counter()
            random_seed = random.randint(0, 1e8)
            
            seg_mask = ants.threshold_image(seg, 0, 0, 0, 1)

            image_patches = list()
            for ii in range(len(images)):
                image_patches.append(antspynet.extract_image_patches(images[ii], 
                                                         patch_size=patch_size,
                                                         max_number_of_patches=number_of_patches_per_image,
                                                         mask_image=seg_mask,
                                                         random_seed=random_seed,
                                                         return_as_array=True))
            if number_of_channels > 1:    
                seg_patches = antspynet.extract_image_patches(seg, 
                                                            patch_size=patch_size,
                                                            max_number_of_patches=number_of_patches_per_image,
                                                            mask_image=seg_mask,
                                                            random_seed=random_seed,
                                                            return_as_array=True)
            labels_patches = antspynet.extract_image_patches(labels, 
                                                          patch_size=patch_size,
                                                          max_number_of_patches=number_of_patches_per_image,
                                                          mask_image=seg_mask,
                                                          random_seed=random_seed,
                                                          return_as_array=True)

            for p in range(number_of_patches_per_image):
                if number_of_channels > 1:
                    seg_patch = np.squeeze(seg_patches[p,:,:,:]) 
                labels_patch = np.squeeze(labels_patches[p,:,:,:]) 

                image_patch = list()            
                for ii in range(len(images)):
                    image_patch.append(np.squeeze(image_patches[ii][p,:,:,:]))
                                             
                number_of_swaps = random.randint(0, 3)
                for _ in range(number_of_swaps):
                    axes = random.sample((0, 1, 2), 2)
                    for ii in range(len(images)):
                        image_patch[ii] = np.swapaxes(image_patch[ii], axes[0], axes[1])
                    if number_of_channels > 1:    
                        seg_patch = np.swapaxes(seg_patch, axes[0], axes[1])
                    labels_patch = np.swapaxes(labels_patch, axes[0], axes[1])
                for axis in range(3):
                    if random.sample((True, False), 1)[0]:
                        for ii in range(len(images)):
                            image_patch[ii] = np.flip(image_patch[ii], axis)
                        if number_of_channels > 1:
                            seg_patch = np.flip(seg_patch, axis) 
                        labels_patch = np.flip(labels_patch, axis) 
                toc = time.perf_counter()
                # print(f"Patches {toc - tic:0.4f} seconds")

                for ii in range(len(images)): 
                    X[batch_count,:,:,:,ii] = image_patch[ii]
                if number_of_channels > 1:
                    for ii in range(3):    
                        seg_patch_one_label = np.zeros(seg_patch.shape)
                        seg_patch_one_label[seg_patch == 1+ii] = 1.0
                        X[batch_count,:,:,:,1+ii] = seg_patch_one_label    
                Y[batch_count,:,:,:,0] = labels_patch
                
                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break
             
        # Yenc = antspynet.encode_unet(Y, (0, 1))     
        yield X, Y, None






