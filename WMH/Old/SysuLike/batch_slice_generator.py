import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    image_size=(64, 64),
                    t1s=None,
                    flairs=None,
                    wmh_images=None,
                    segmentation_images=None,
                    use_t1s=True,
                    use_flairs=True,
                    use_segmentation_images=False,
                    number_of_slices_per_image=5,
                    do_random_contralateral_flips=True,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_data_augmentation=True,
                    use_rank_intensity_scaling=False):

    if t1s is None or flairs is None:
        raise ValueError("Input images must be specified.")

    if wmh_images is None or segmentation_images is None:
        raise ValueError("Input masks must be specified.")

    while True:

        number_of_channels = 0
        if use_t1s == True:
            number_of_channels += 1
        if use_flairs == True:
            number_of_channels += 1
        if use_segmentation_images == True:
            number_of_channels += 1

        X = np.zeros((batch_size, *image_size, number_of_channels))
        Y  = np.zeros((batch_size, *image_size))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(segmentation_images))), 1)[0]

            flair = ants.image_read(flairs[i])
            t1 = ants.image_read(t1s[i])
            seg = ants.image_read(segmentation_images[i])
            wmh = ants.threshold_image(ants.image_read(wmh_images[i]), 1, 1, 1, 0)

            t1 = ants.copy_image_info(flair, t1)
            seg = ants.copy_image_info(flair, seg)
            wmh = ants.copy_image_info(flair, wmh)

            geoms = ants.label_geometry_measures(wmh)
            if len(geoms['Label']) == 0:
                continue

            resampling_params = np.array(ants.get_spacing(flair))

            do_resampling = False
            for d in range(len(resampling_params)):
                if resampling_params[d] < 0.8:
                    resampling_params[d] = 1.0
                    do_resampling = True

            resampling_params = tuple(resampling_params)

            if do_resampling:
                t1 = ants.resample_image(t1, resampling_params, use_voxels=False, interp_type=0)
                flair = ants.resample_image(flair, resampling_params, use_voxels=False, interp_type=0)
                seg = ants.resample_image(seg, resampling_params, use_voxels=False, interp_type=1)
                wmh = ants.resample_image(wmh, resampling_params, use_voxels=False, interp_type=1)

            t1 = antspynet.pad_or_crop_image_to_size(t1, (*image_size, t1.shape[2]))
            flair = antspynet.pad_or_crop_image_to_size(flair, (*image_size, t1.shape[2]))
            seg = antspynet.pad_or_crop_image_to_size(seg, (*image_size, t1.shape[2]))
            wmh = antspynet.pad_or_crop_image_to_size(wmh, (*image_size, t1.shape[2]))            

            # if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
            #     t1 = ants.reflect_image(t1, axis=0)
            #     flair = ants.reflect_image(flair, axis=0)
            #     seg = ants.reflect_image(seg, axis=0)
            #     wmh = ants.reflect_image(wmh, axis=0)

            if do_histogram_intensity_warping and random.sample((True, False), 1)[0]:
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.05)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                t1 = antspynet.histogram_warp_image_intensities(t1,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)
                flair = antspynet.histogram_warp_image_intensities(flair,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)

            if do_add_noise and random.sample((True, False), 1)[0]:
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())
                flair = (flair - flair.min()) / (flair.max() - flair.min())
                noise_parameters = (0.0, random.uniform(0, 0.05))
                t1 = ants.add_noise_to_image(t1, noise_model="additivegaussian", noise_parameters=noise_parameters)
                flair = ants.add_noise_to_image(flair, noise_model="additivegaussian", noise_parameters=noise_parameters)

            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                t1_field = antspynet.simulate_bias_field(t1, sd_bias_field=0.05) 
                flair_field = antspynet.simulate_bias_field(flair, sd_bias_field=0.05)
                t1 = ants.iMath(t1, "Normalize") * (t1_field + 1)
                flair = ants.iMath(flair, "Normalize") * (flair_field + 1)
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())
                flair = (flair - flair.min()) / (flair.max() - flair.min())                

            t1_pre = antspynet.preprocess_brain_image(t1, truncate_intensity=(0.01, 0.995), do_bias_correction=False, do_denoising=False, verbose=False)
            flair_pre = antspynet.preprocess_brain_image(flair, truncate_intensity=(0.01, 0.995), do_bias_correction=False, do_denoising=False, verbose=False)

            t1 = t1_pre['preprocessed_image']
            flair = flair_pre['preprocessed_image']
            mask = ants.threshold_image(seg, 0, 0, 0, 1)

            if use_rank_intensity_scaling:
                t1 = ants.rank_intensity(t1, mask) - 0.5
                flair = ants.rank_intensity(flair, mask) - 0.5
            else:
                t1 = (t1 - t1[seg != 0].mean()) / t1[seg != 0].std()
                flair = (flair - flair[seg != 0].mean()) / flair[seg != 0].std()

            t1 = t1 * mask
            flair = flair * mask

            wmh_array = wmh.numpy()
            t1_array = t1.numpy()
            flair_array = flair.numpy()
            seg_array = seg.numpy()

            which_dimension_max_spacing = resampling_params.index(max(resampling_params))
            if which_dimension_max_spacing == 0:
                lower_slice = geoms['BoundingBoxLower_x'][0]
                upper_slice = geoms['BoundingBoxUpper_x'][0]
            elif which_dimension_max_spacing == 1:
                lower_slice = geoms['BoundingBoxLower_y'][0]
                upper_slice = geoms['BoundingBoxUpper_y'][0]
            else:
                lower_slice = geoms['BoundingBoxLower_z'][0]
                upper_slice = geoms['BoundingBoxUpper_z'][0]
            if lower_slice >= upper_slice:
                continue

            number_of_samples = min(number_of_slices_per_image, upper_slice - lower_slice)
            if number_of_samples <= 0:
                continue

            which_random_slices = random.sample(list(range(lower_slice, upper_slice)), number_of_samples)

            for j in range(len(which_random_slices)):
                which_slice = which_random_slices[j]

                t1_slice = None
                flair_slice = None
                wmh_slice = None
                seg_slice = None

                if which_dimension_max_spacing == 0:
                    t1_slice = ants.from_numpy(np.squeeze(t1_array[which_slice,:,:]))
                    flair_slice = ants.from_numpy(np.squeeze(flair_array[which_slice,:,:]))
                    wmh_slice = ants.from_numpy(np.squeeze(wmh_array[which_slice,:,:]))
                    seg_slice = ants.from_numpy(np.squeeze(seg_array[which_slice,:,:]))
                elif which_dimension_max_spacing == 1:
                    t1_slice = ants.from_numpy(np.squeeze(t1_array[:,which_slice,:]))
                    flair_slice = ants.from_numpy(np.squeeze(flair_array[:,which_slice,:]))
                    wmh_slice = ants.from_numpy(np.squeeze(wmh_array[:,which_slice,:]))
                    seg_slice = ants.from_numpy(np.squeeze(seg_array[:,which_slice,:]))
                else:
                    t1_slice = ants.from_numpy(np.squeeze(t1_array[:,:,which_slice]))
                    flair_slice = ants.from_numpy(np.squeeze(flair_array[:,:,which_slice]))
                    wmh_slice = ants.from_numpy(np.squeeze(wmh_array[:,:,which_slice]))
                    seg_slice = ants.from_numpy(np.squeeze(seg_array[:,:,which_slice]))

                # Ensure that only ~10% are empty of any wmhs (i.e., voxel count < 100)
#                 if wmh_slice.sum() < 500: # and random.uniform(0.0, 1) > 0.1:
#                    continue

                if do_data_augmentation == True:
                    data_augmentation = antspynet.randomly_transform_image_data(seg_slice,
                        [[t1_slice, flair_slice]],
                       #  [wmh_slice],
                        number_of_simulations=1,
                        transform_type='affineAndDeformation',
                        sd_affine=0.01,
                        deformation_transform_type="bspline",
                        number_of_random_points=1000,
                        sd_noise=2.0,
                        number_of_fitting_levels=4,
                        mesh_size=1,
                        sd_smoothing=4.0,
                        input_image_interpolator='linear',
                        segmentation_image_interpolator='nearestNeighbor')

                    t1_slice = data_augmentation['simulated_images'][0][0]
                    flair_slice = data_augmentation['simulated_images'][0][1]
                    # wmh_slice = data_augmentation['simulated_segmentation_images'][0]
                    # seg_slice = data_augmentation['simulated_segmentation_images'][0]
                    wmh_slice = ants.apply_ants_transform_to_image(
                       data_augmentation['simulated_transforms'][0], wmh_slice, t1_slice,
                       interpolation="linear")
                    seg_slice = ants.apply_ants_transform_to_image(
                       data_augmentation['simulated_transforms'][0], seg_slice, t1_slice,
                       interpolation="nearestneighbor")

                t1_slice = antspynet.pad_or_crop_image_to_size(t1_slice, image_size)
                flair_slice = antspynet.pad_or_crop_image_to_size(flair_slice, image_size)
                wmh_slice = antspynet.pad_or_crop_image_to_size(wmh_slice, image_size)
                seg_slice = antspynet.pad_or_crop_image_to_size(seg_slice, image_size)

                # brain extraction
                flair_slice[seg_slice == 0] = 0
                t1_slice[seg_slice == 0] = 0

                t1_slice_array = t1_slice.numpy()
                flair_slice_array = flair_slice.numpy()
                seg_slice_array = seg_slice.numpy()
                wmh_slice_array = wmh_slice.numpy()
                if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
                    t1_slice_array = np.flip(t1_slice_array, axis=0)
                    flair_slice_array = np.flip(flair_slice_array, axis=0)
                    seg_slice_array = np.flip(seg_slice_array, axis=0)
                    wmh_slice_array = np.flip(wmh_slice_array, axis=0)

                wmh_slice_array[wmh_slice_array >= 0.5] = 1
                wmh_slice_array[wmh_slice_array < 0.5] = 0

               # ants.image_write(ants.from_numpy(t1_slice_array), "t1" + str(batch_count) + ".nii.gz")
               # ants.image_write(ants.from_numpy(flair_slice_array), "flair" + str(batch_count) + ".nii.gz")
               # ants.image_write(ants.from_numpy(seg_slice_array), "seg" + str(batch_count) + ".nii.gz")           
               # ants.image_write(ants.from_numpy(wmh_slice_array), "wmh" + str(batch_count) + ".nii.gz")                

                channel_count = 0
                if use_flairs == True:
                    X[batch_count,:,:,channel_count] = flair_slice_array
                    channel_count += 1
                if use_t1s == True:
                    X[batch_count,:,:,channel_count] = t1_slice_array
                    channel_count += 1
                if use_segmentation_images == True:    
                    X[batch_count,:,:,channel_count] = seg_slice_array / 6 - 0.5
                    channel_count += 1

                Y[batch_count,:,:] = wmh_slice_array

                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break

            if batch_count >= batch_size:
                break

        # raise ValueError("HERE")

        yield X, Y, [None]









