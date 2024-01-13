import numpy as np
import random
import ants
import antspynet
import os

import time

def batch_generator(batch_size=32,
                    t2_files=None,
                    t1_files=None,
                    atropos_files=None,
                    sysu_files=None,
                    bianca_files=None,
                    patch_size=(64, 64, 64),
                    number_of_channels=2,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=False):

    if t2_files is None or t1_files is None or atropos_files is None or sysu_files is None or bianca_files is None:
        raise ValueError("Input images must be specified.")

    number_of_patches_per_image = 4
    wmh_volume_per_patch = 0

    while True:

        X = np.zeros((batch_size, *patch_size, number_of_channels))
        Y  = np.zeros((batch_size, *patch_size, 1))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(t1_files))), 1)[0]

            t1 = ants.image_read(t1_files[i])
            t2 = ants.image_read(t2_files[i])
            atropos = ants.image_read(atropos_files[i])
            sysu = ants.image_read(sysu_files[i])
            bianca = ants.image_read(bianca_files[i])
            
            # t1 = antspynet.pad_or_crop_image_to_size(t1, (250, 250, 250))
            # t2 = antspynet.pad_or_crop_image_to_size(t2, (250, 250, 250))
            # atropos = antspynet.pad_or_crop_image_to_size(atropos, (250, 250, 250))
            # sysu = antspynet.pad_or_crop_image_to_size(sysu, (250, 250, 250))
            # bianca = antspynet.pad_or_crop_image_to_size(bianca, (250, 250, 250))

            if do_histogram_intensity_warping: # and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.05)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                t1 = antspynet.histogram_warp_image_intensities(t1,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.05)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                t2 = antspynet.histogram_warp_image_intensities(t2,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)
                toc = time.perf_counter()
                # print(f"Histogram warping {toc - tic:0.4f} seconds")

            if do_add_noise and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())
                noise_parameters = (0.0, random.uniform(0, 0.01))
                t1 = ants.add_noise_to_image(t1, noise_model="additivegaussian", noise_parameters=noise_parameters)
                t2 = (t2 - t2.min()) / (t2.max() - t2.min())
                noise_parameters = (0.0, random.uniform(0, 0.01))
                t2 = ants.add_noise_to_image(t2, noise_model="additivegaussian", noise_parameters=noise_parameters)
                toc = time.perf_counter()
                # print(f"Add noise {toc - tic:0.4f} seconds")

            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                log_field = antspynet.simulate_bias_field(t1, number_of_points=10, sd_bias_field=1.0, number_of_fitting_levels=2, mesh_size=10)
                log_field = log_field.iMath("Normalize")
                field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3, 4), 1)[0])
                t1 = t1 * ants.from_numpy(field_array, origin=t1.origin, spacing=t1.spacing, direction=t1.direction)
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())

                log_field = antspynet.simulate_bias_field(t2, number_of_points=10, sd_bias_field=1.0, number_of_fitting_levels=2, mesh_size=10)
                log_field = log_field.iMath("Normalize")
                field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3, 4), 1)[0])
                t2 = t2 * ants.from_numpy(field_array, origin=t2.origin, spacing=t2.spacing, direction=t2.direction)
                t2 = (t2 - t2.min()) / (t2.max() - t2.min())
                toc = time.perf_counter()
                # print(f"Sim bias field {toc - tic:0.4f} seconds")

            if do_random_transformation:
                tic = time.perf_counter()
                data_augmentation = antspynet.randomly_transform_image_data(t1,
                    [[t1, t2]],
                    [atropos],
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
                sysu = ants.apply_ants_transform_to_image(data_augmentation['simulated_transforms'][0], sysu,
                    reference=t1, interpolation='nearestneighbor')
                bianca = ants.apply_ants_transform_to_image(data_augmentation['simulated_transforms'][0], bianca,
                    reference=t1, interpolation='nearestneighbor')
                t1 = data_augmentation['simulated_images'][0][0]
                t2 = data_augmentation['simulated_images'][0][1]
                atropos = data_augmentation['simulated_segmentation_images'][0]
                toc = time.perf_counter()
                # print(f"Xfrms {toc - tic:0.4f} seconds")

            tic = time.perf_counter()
            # wmh_field = antspynet.simulate_bias_field(sysu, number_of_points=10, sd_bias_field=1.0, number_of_fitting_levels=2, mesh_size=10)
            # wmh_field = wmh_field.iMath("Normalize") * 0.5 + 0.25
            # ones_field = ants.image_clone(sysu) * 0 + 1
            # wmh = sysu * wmh_field + bianca * (ones_field - wmh_field)
            wmh = bianca
            # wmh = sysu + bianca - (sysu * bianca)

            brain_mask = ants.threshold_image(atropos, 0, 0, 0, 1)
            t1 = t1 * brain_mask
            t2 = t2 * brain_mask
            wm_mask = ants.threshold_image(atropos, 3, 4, 1, 0)
            # wmh = wmh * wm_mask

            indices = wm_mask > 0
            t1 = (t1 - t1[indices].min()) / (t1[indices].max() - t1[indices].min())
            t2 = (t2 - t2[indices].min()) / (t2[indices].max() - t2[indices].min())
            toc = time.perf_counter()
            # print(f"Misc {toc - tic:0.4f} seconds")


            # ants.image_write(t2, "image_t2_" + str(batch_count) + ".nii.gz")
            # ants.image_write(t1, "image_t1_" + str(batch_count) + ".nii.gz")


            tic = time.perf_counter()
            random_seed = random.randint(0, 1e8)
            
            t1_patches = antspynet.extract_image_patches(t1, 
                                                         patch_size=patch_size,
                                                         max_number_of_patches=number_of_patches_per_image,
                                                         mask_image=wm_mask,
                                                         random_seed=random_seed,
                                                         return_as_array=True)
            t2_patches = antspynet.extract_image_patches(t2, 
                                                         patch_size=patch_size,
                                                         max_number_of_patches=number_of_patches_per_image,
                                                         mask_image=wm_mask,
                                                         random_seed=random_seed,
                                                         return_as_array=True)
            wm_mask_patches = antspynet.extract_image_patches(wm_mask, 
                                                              patch_size=patch_size,
                                                              max_number_of_patches=number_of_patches_per_image,
                                                              mask_image=wm_mask,
                                                              random_seed=random_seed,
                                                              return_as_array=True)
            wmh_patches = antspynet.extract_image_patches(wmh, 
                                                          patch_size=patch_size,
                                                          max_number_of_patches=number_of_patches_per_image,
                                                          mask_image=wm_mask,
                                                          random_seed=random_seed,
                                                          return_as_array=True)
            
            for p in range(number_of_patches_per_image):
                wmh_patch = np.squeeze(wmh_patches[p,:,:,:]) 
                if wmh_patch.sum() < wmh_volume_per_patch:
                    continue

                t2_patch = np.squeeze(t2_patches[p,:,:,:]) 
                t1_patch = np.squeeze(t1_patches[p,:,:,:]) 
                wm_mask_patch = np.squeeze(wm_mask_patches[p,:,:,:])
                                             
                number_of_swaps = random.randint(0, 3)
                for _ in range(number_of_swaps):
                    axes = random.sample((0, 1, 2), 2)
                    t2_patch = np.swapaxes(t2_patch, axes[0], axes[1]) 
                    t1_patch = np.swapaxes(t1_patch, axes[0], axes[1]) 
                    wm_mask_patch = np.swapaxes(wm_mask_patch, axes[0], axes[1]) 
                    wmh_patch = np.swapaxes(wmh_patch, axes[0], axes[1])
                for axis in range(3):
                    if random.sample((True, False), 1)[0]:
                        t2_patch = np.flip(t2_patch, axis) 
                        t1_patch = np.flip(t1_patch, axis) 
                        wm_mask_patch = np.flip(wm_mask_patch, axis) 
                        wmh_patch = np.flip(wmh_patch, axis) 
                toc = time.perf_counter()
                # print(f"Patches {toc - tic:0.4f} seconds")
                    
                X[batch_count,:,:,:,0] = t2_patch
                X[batch_count,:,:,:,1] = t1_patch
                # X[batch_count,:,:,:,2] = wm_mask_patch
                Y[batch_count,:,:,:,0] = wmh_patch
                # Y[batch_count,:,:,:,0] = np.where(wmh_patch >= 0.5, 1, 0)
                # encY[batch_count,:,:,:,0] = np.ones(wmh_patches.shape) - wmh_patches
                # encY[batch_count,:,:,:,1] = wmh_patches

                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break

        yield X, Y, None






