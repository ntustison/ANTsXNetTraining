import numpy as np
import random
import ants
import antspynet
import time

def batch_generator(batch_size,
                    input_image_files,
                    image_size,
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

    X = np.zeros((batch_size, *image_size))
    # X = np.zeros((batch_size, np.prod(image_size)))

    while True:

        batch_count = 0

        while batch_count < batch_size:

            r_idx = random.sample(list(range(len(input_image_files))), 1)[0]

            if verbose:
                print("Batch count: ", batch_count)

            images = list()
            for i in range(1):
                image = ants.image_read(input_image_files[r_idx])
                image = image.iMath("Normalize")
                if do_resampling:
                    resample_spacing = np.random.uniform(low=0.7, high=1.2, size=3) 
                    downsampled_image = ants.resample_image(image, resample_params=resample_spacing,
                                                            use_voxels=False, interp_type=0)
                    image = ants.resample_image_to_target(downsampled_image, image)
                images.append(image)

            if do_random_transformation:
                tic = time.perf_counter()
                data_augmentation = antspynet.randomly_transform_image_data(images[0],
                    [images],
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
                toc = time.perf_counter()
                if verbose:
                    print(f"    Do random transformation {toc - tic:0.4f} seconds.")

            if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                for i in range(len(images)):
                    image_array = images[i].numpy()
                    image_array = np.flip(image_array, axis=0)
                    images[i] = ants.from_numpy_like(image_array, images[i])
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
                image = ants.rank_intensity(image)
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
             
            # X_img_array = np.zeros((*image_size, len(images))) 
            # for i in range(len(images)):                
            #     images[i] = ants.iMath_normalize(images[i])
            #     X_img_array[:,:,i] = np.expand_dims(images[i].numpy(), axis=-1)
            # X[batch_count,:] = X_img_array.flatten()

            for i in range(len(images)):                
                images[i] = ants.iMath_normalize(images[i])
                X[batch_count,:,:,i] = images[i].numpy()
            batch_count += 1    
                                
            if batch_count >= batch_size:
                break

        yield X

