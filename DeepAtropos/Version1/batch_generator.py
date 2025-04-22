import numpy as np
import random
import ants
import antspynet
import time

def batch_generator(batch_size,
                    template,
                    priors,
                    template_modalities,
                    template_segmentation,
                    template_brain_mask,
                    patch_size=(64, 64, 64),
                    number_of_octants_per_image=2,
                    do_histogram_intensity_warping=True,
                    do_histogram_equalization=True,
                    do_histogram_rank_intensity=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=True,
                    do_random_contralateral_flips=True,
                    do_resampling=True,
                    verbose=False
                    ):


    stride_length = tuple(np.array(template[0].shape) - np.array(patch_size))

    channel_size = len(template_modalities) + len(priors)
    
    number_of_templates = len(template)

    X = np.zeros((batch_size, *patch_size, channel_size))
    Y  = np.zeros((batch_size, *patch_size))

    priors_patches = list()
    for i in range(len(priors)):
        priors_patches.append( 
            antspynet.extract_image_patches(priors[i], patch_size, max_number_of_patches="all",
                stride_length=stride_length, random_seed=None, return_as_array=True)
            )

    while True:

        batch_count = 0

        while batch_count < batch_size:

            t_idx = random.sample(list(range(number_of_templates)), 1)[0]


            if verbose:
                print("Batch count: ", batch_count)


            images = list()
            segmentation = ants.image_clone(template_segmentation[t_idx])                
            for i in range(len(template_modalities)):
                image = ants.image_clone(template_modalities[i][t_idx]) * template_brain_mask[t_idx]
                image = image.iMath("Normalize")
                if do_resampling:
                    resample_spacing = np.random.uniform(low=0.7, high=1.2, size=3) 
                    downsampled_image = ants.resample_image(image, resample_params=resample_spacing,
                                                            use_voxels=False, interp_type=0)
                    image = ants.resample_image_to_target(downsampled_image, image)
                images.append(image)

            if do_random_transformation:
                tic = time.perf_counter()
                data_augmentation = antspynet.randomly_transform_image_data(template[t_idx],
                    [images],
                    [segmentation],
                    number_of_simulations=1,
                    transform_type='deformation',
                    sd_affine=0.01,
                    deformation_transform_type="bspline",
                    number_of_random_points=1000,
                    sd_noise=10.0,
                    number_of_fitting_levels=4,
                    mesh_size=1,
                    sd_smoothing=4.0,
                    input_image_interpolator='linear',
                    segmentation_image_interpolator='nearestNeighbor')
                images = data_augmentation['simulated_images'][0]
                segmentation = data_augmentation['simulated_segmentation_images'][0]
                toc = time.perf_counter()
                if verbose:
                    print(f"    Do random transformation {toc - tic:0.4f} seconds.")

            if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                for i in range(len(images)):
                    image_array = images[i].numpy()
                    image_array = np.flip(image_array, axis=0)
                    images[i] = ants.from_numpy_like(image_array, images[i])
                segmentation_array = segmentation.numpy()
                segmentation_array = np.flip(segmentation_array, axis=0)
                segmentation = ants.from_numpy_like(segmentation_array, segmentation)
                toc = time.perf_counter()
                if verbose:
                    print(f"    Random contralateral flip {toc - tic:0.4f} seconds")

            if do_histogram_intensity_warping: # and random.sample((True, True, False), 1)[0]:
                tic = time.perf_counter()
                for i in range(len(images)):
                    break_points = [0.2, 0.4, 0.6, 0.8]
                    displacements = list()
                    for b in range(len(break_points)):
                        displacements.append(abs(random.gauss(0, 0.05)))
                        if random.sample((True, False), 1)[0]:
                            displacements[b] *= -1
                    images[i] = antspynet.histogram_warp_image_intensities(images[i],
                        break_points=break_points, clamp_end_points=(True, False),
                        displacements=displacements)
                    images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
                toc = time.perf_counter()
                if verbose:
                    print(f"    Histogram warping {toc - tic:0.4f} seconds")

            if do_histogram_equalization and random.sample((False, False, True), 1)[0]:
                tic = time.perf_counter()
                image = ants.histogram_equalize_image(image)
                image = image.iMath("Normalize")
                toc = time.perf_counter()
                if verbose:
                    print(f"    Histogram equalization {toc - tic:0.4f} seconds")

            if do_histogram_rank_intensity and random.sample((False, False, True), 1)[0]:
                tic = time.perf_counter()
                image = ants.rank_intensity(image, ants.threshold_image(segmentation, 0, 0, 0, 1))
                image = image.iMath("Normalize")
                toc = time.perf_counter()
                if verbose:
                    print(f"    Histogram rank intensity {toc - tic:0.4f} seconds")

            if do_simulate_bias_field and random.sample((True, True, False), 1)[0]:
                tic = time.perf_counter()
                for i in range(len(images)):
                    log_field = antspynet.simulate_bias_field(images[i], 
                                                              number_of_points=100, 
                                                              sd_bias_field=0.1, 
                                                              number_of_fitting_levels=2, 
                                                              mesh_size=10)
                    log_field = log_field.iMath("Normalize")
                    field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3), 1)[0])
                    images[i] = images[i] * ants.from_numpy_like(field_array, images[i])
                    images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
                toc = time.perf_counter()
                if verbose:
                    print(f"    Simulate bias field {toc - tic:0.4f} seconds")

            if do_add_noise and random.sample((True, True, False), 1)[0]:
                tic = time.perf_counter()
                for i in range(len(images)):
                    image_ones = images[i] * 0 + 1                
                    noise_parameters = (0.1, random.uniform(0.05, 0.15))
                    noise = ants.add_noise_to_image(image_ones, noise_model="additivegaussian", noise_parameters=noise_parameters)
                    noise = ants.smooth_image(noise, sigma=random.uniform(0.25, 0.5))
                    images[i] = images[i] * noise
                    images[i] = images[i].iMath("Normalize")
                toc = time.perf_counter()
                if verbose:
                    print(f"    Add noise {toc - tic:0.4f} seconds")

            for i in range(len(images)):                
                images[i] = ants.iMath_normalize(images[i]) 

            # image_brain_mask = ants.threshold_image(segmentation, 0, 0, 0, 1)
            # for i in range(len(images)):                
            #     images[i] = ants.histogram_match_image2(images[i], 
            #                                             template_modalities[i][t_idx],
            #                                             image_brain_mask,
            #                                             template_brain_mask[t_idx])
                
            tic = time.perf_counter()

            image_patches_list = list()            
            for i in range(len(images)):                
                image_patches = antspynet.extract_image_patches(images[i], 
                                                                patch_size, 
                                                                max_number_of_patches="all",
                                                                stride_length=stride_length, 
                                                                random_seed=None, 
                                                                return_as_array=True)
                image_patches_list.append(image_patches)    
            segmentation_patches = antspynet.extract_image_patches(segmentation, 
                                                            patch_size, 
                                                            max_number_of_patches="all",
                                                            stride_length=stride_length, 
                                                            random_seed=None, 
                                                            return_as_array=True)

            # octant 0/4 are the cerebellum
            # which_octants = random.sample(list(range(8)), counts=[3, 1, 1, 1, 3, 1, 1, 1],
            #                               k=number_of_octants_per_image)
            which_octants = random.sample(list(range(8)), counts=[1, 1, 1, 1, 1, 1, 1, 1],
                                          k=number_of_octants_per_image)

            for o in range(number_of_octants_per_image):  
                for i in range(len(images)):                
                    X[batch_count,:,:,:,i] = image_patches_list[i][which_octants[o],:,:,:] 
                for p in range(len(priors_patches)):
                    X[batch_count,:,:,:,len(images) + p] = priors_patches[p][which_octants[o],:,:,:]
                Y[batch_count,:,:,:] = segmentation_patches[which_octants[o],:,:,:]

                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break

            toc = time.perf_counter()
            if verbose:
                print(f"    Patchify {toc - tic:0.4f} seconds.")
                
            if batch_count >= batch_size:
                break

        tic = time.perf_counter()
        encoded_Y = antspynet.encode_unet(Y, list(range(7)))
        toc = time.perf_counter()
        if verbose:
            print(f"Encode Y {toc - tic:0.4f} seconds.")

        if verbose:
            yield X, Y
        else:
            yield X, encoded_Y

