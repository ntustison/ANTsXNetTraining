import numpy as np
import random
import ants
import antspynet
import time

def batch_generator(batch_size,
                    template,
                    training_image_files,
                    training_segmentation_files,
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

    X = np.zeros((batch_size, *template.shape, 1))
    Y  = np.zeros((batch_size, *template.shape))

    while True:

        batch_count = 0

        while batch_count < batch_size:

            t_idx = random.sample(list(range(len(training_image_files))), 1)[0]

            if verbose:
                print("Batch count: ", batch_count)

            image = ants.image_read(training_image_files[t_idx])
            segmentation = ants.image_read(training_segmentation_files[t_idx])

            center_of_mass_template = ants.get_center_of_mass(template * 0 + 1)
            center_of_mass_image = ants.get_center_of_mass(image * 0 + 1)
            translation = np.round(np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template))
            xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
                center=np.round(np.asarray(center_of_mass_template)), translation=translation)            
            image = ants.apply_ants_transform_to_image(xfrm, image=image, reference=template, interpolation="linear")
            segmentation = ants.apply_ants_transform_to_image(xfrm, image=segmentation, reference=template, interpolation="nearestNeighbor")

            image = (image - image.min()) / (image.max() - image.min())

            if do_random_transformation:
                tic = time.perf_counter()
                data_augmentation = antspynet.randomly_transform_image_data(template,
                    [[image]],
                    [segmentation],
                    number_of_simulations=1,
                    transform_type='affineAndDeformation',
                    sd_affine=0.075,
                    deformation_transform_type="bspline",
                    number_of_random_points=1000,
                    sd_noise=10.0,
                    number_of_fitting_levels=4,
                    mesh_size=1,
                    sd_smoothing=4.0,
                    input_image_interpolator='linear',
                    segmentation_image_interpolator='nearestNeighbor')
                image = data_augmentation['simulated_images'][0][0]
                segmentation = data_augmentation['simulated_segmentation_images'][0]
                toc = time.perf_counter()
                if verbose:
                    print(f"    Do random transformation {toc - tic:0.4f} seconds.")

            if do_resampling:
                resample_spacing = np.random.uniform(low=1.0, high=1.5, size=3) 
                downsampled_image = ants.resample_image(image, resample_params=resample_spacing,
                                                        use_voxels=False, interp_type=0)
                image = ants.resample_image_to_target(downsampled_image, image)
 
            if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                image_array = image.numpy()
                image_array = np.flip(image_array, axis=0)
                image = ants.from_numpy_like(image_array, image)
                segmentation[segmentation == 1] = 3
                segmentation[segmentation == 2] = 1
                segmentation[segmentation == 3] = 2
                segmentation_array = segmentation.numpy()
                segmentation_array = np.flip(segmentation_array, axis=0)
                segmentation = ants.from_numpy_like(segmentation_array, segmentation)
                toc = time.perf_counter()
                if verbose:
                    print(f"    Random contralateral flip {toc - tic:0.4f} seconds")

            if do_histogram_intensity_warping and random.sample((True, True, False), 1)[0]:
                tic = time.perf_counter()
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.10)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                image = antspynet.histogram_warp_image_intensities(image,
                    break_points=break_points, clamp_end_points=(False, False),
                    displacements=displacements)
                image = image.iMath("Normalize")
                toc = time.perf_counter()
                if verbose:
                    print(f"    Histogram warping {toc - tic:0.4f} seconds")

            if do_histogram_equalization and random.sample((False, True), 1)[0]:
                tic = time.perf_counter()
                image = ants.histogram_equalize_image(image)
                image = image.iMath("Normalize")
                toc = time.perf_counter()
                if verbose:
                    print(f"    Histogram equalization {toc - tic:0.4f} seconds")

            if do_histogram_rank_intensity and random.sample((False, True), 1)[0]:
                tic = time.perf_counter()
                image = ants.rank_intensity(image, ants.threshold_image(segmentation, 0, 0, 0, 1))
                image = image.iMath("Normalize")
                toc = time.perf_counter()
                if verbose:
                    print(f"    Histogram rank intensity {toc - tic:0.4f} seconds")

            if do_simulate_bias_field and random.sample((True, True, False), 1)[0]:
                tic = time.perf_counter()
                log_field = antspynet.simulate_bias_field(image, 
                                                            number_of_points=100, 
                                                            sd_bias_field=0.1, 
                                                            number_of_fitting_levels=2, 
                                                            mesh_size=10)
                log_field = log_field.iMath("Normalize")
                field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3), 1)[0])
                image = image * ants.from_numpy_like(field_array, image)
                image = image.iMath("Normalize")
                toc = time.perf_counter()
                if verbose:
                    print(f"    Simulate bias field {toc - tic:0.4f} seconds")

            if do_add_noise and random.sample((True, True, False), 1)[0]:
                tic = time.perf_counter()
                image_ones = image * 0 + 1                
                noise_parameters = (0.1, random.uniform(0.05, 0.15))
                noise = ants.add_noise_to_image(image_ones, noise_model="additivegaussian", noise_parameters=noise_parameters)
                noise = ants.smooth_image(noise, sigma=random.uniform(0.25, 0.5))
                image = image * noise
                image = image.iMath("Normalize")
                toc = time.perf_counter()
                if verbose:
                    print(f"    Add noise {toc - tic:0.4f} seconds")

            image = ants.iMath_normalize(image) 

            tic = time.perf_counter()

            X[batch_count,:,:,:,0] = image.numpy()
            Y[batch_count,:,:,:] = segmentation.numpy()
                
            batch_count = batch_count + 1    
            if batch_count >= batch_size:
                break

        tic = time.perf_counter()
        encoded_Y = antspynet.encode_unet(Y, np.unique(Y).astype('int32'))
        toc = time.perf_counter()
        if verbose:
            print(f"Encode Y {toc - tic:0.4f} seconds.")

        if verbose:
            yield X, Y
        else:
            yield X, encoded_Y

