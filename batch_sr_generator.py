import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    lr_images=None,
                    sr_images=None,
                    do_smoothing=True,
                    do_random_contralateral_flips=True,
                    do_simulate_bias_field=True,
                    do_histogram_intensity_warping=True,
                    do_add_noise=True):

    if sr_images is None or lr_images is None:
        raise ValueError("Input images must be specified.")

    lr_image_size = (256, 256, 3)
    sr_image_size = (512, 512, 3)

    while True:

        X = np.zeros((batch_size, *lr_image_size))
        Y = np.zeros((batch_size, *sr_image_size))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(lr_images))), 1)[0]

            lr_image_rgb = ants.image_read(lr_images[i], dimension=2)
            sr_image_rgb = ants.image_read(sr_images[i], dimension=2)
            
            lr_image_channels = ants.split_channels(lr_image_rgb)
            sr_image_channels = ants.split_channels(sr_image_rgb)
            
            smoothing_parameter = random.sample((0, 1, 2), 1)[0]

            for c in range(len(lr_image_channels)):
                lr_image = lr_image_channels[c]
                sr_image = sr_image_channels[c]

                lr_image = (lr_image - lr_image.min()) / (lr_image.max() - lr_image.min()) * -1 + 1
                sr_image = (sr_image - sr_image.min()) / (sr_image.max() - sr_image.min()) * -1 + 1

                if do_smoothing and smoothing_parameter > 0:
                    lr_image = ants.smooth_image(lr_image, sigma=smoothing_parameter, sigma_in_physical_coordinates=False, FWHM=False)

                if do_histogram_intensity_warping and random.sample((True, False), 1)[0]:
                    break_points = [0.2, 0.4, 0.6, 0.8]
                    displacements = list()
                    for b in range(len(break_points)):
                        displacements.append(abs(random.gauss(0, 0.175)))
                        if random.sample((True, False), 1)[0]:
                            displacements[b] *= -1
                    lr_image = antspynet.histogram_warp_image_intensities(lr_image,
                        break_points=break_points, clamp_end_points=(True, True),
                        displacements=displacements)
                    sr_image = antspynet.histogram_warp_image_intensities(sr_image,
                        break_points=break_points, clamp_end_points=(True, True),
                        displacements=displacements)

                if do_add_noise and random.sample((True, False), 1)[0]:
                    lr_image = (lr_image - lr_image.min()) / (lr_image.max() - lr_image.min())
                    sr_image = (sr_image - sr_image.min()) / (sr_image.max() - sr_image.min())
                    noise_parameters = (0.0, random.uniform(0, 0.01))
                    lr_image = ants.add_noise_to_image(lr_image, noise_model="additivegaussian", noise_parameters=noise_parameters)
                    sr_image = ants.add_noise_to_image(sr_image, noise_model="additivegaussian", noise_parameters=noise_parameters)

                if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                    log_field = antspynet.simulate_bias_field(sr_image, number_of_points=10, sd_bias_field=1.0,
                                number_of_fitting_levels=2, mesh_size=10)
                    log_field = log_field.iMath("Normalize")
                    log_field_array = np.power(np.exp(log_field.numpy()), random.sample((3, 4, 5), 1)[0])
                    log_field_image = ants.from_numpy(log_field_array, origin=sr_image.origin,
                                                    spacing=sr_image.spacing, direction=sr_image.direction)          
                    sr_image = sr_image * log_field_image
                    log_field_image = ants.resample_to_target(log_field_image, lr_image)
                    lr_image = lr_image * log_field_image

                sr_image_array = (sr_image.numpy()).astype('float64')
                lr_image_array = (lr_image.numpy()).astype('float64')

                if do_random_contralateral_flips:
                    if random.sample((True, False), 1)[0]:
                        lr_image_array = np.fliplr(lr_image_array)
                        sr_image_array = np.fliplr(sr_image_array)
                    if random.sample((True, False), 1)[0]:
                        lr_image_array = np.flipud(lr_image_array)
                        sr_image_array = np.flipud(sr_image_array)
                    if random.sample((True, False), 1)[0]:
                        number_of_rot90s = random.sample((1, 2, 3), 1)[0]
                        lr_image_array = np.rot90(lr_image_array, k=number_of_rot90s)
                        sr_image_array = np.rot90(sr_image_array, k=number_of_rot90s)

                lr_image_array = (lr_image_array - lr_image_array.min()) / (lr_image_array.max() - lr_image_array.min())
                sr_image_array = (sr_image_array - sr_image_array.min()) / (sr_image_array.max() - sr_image_array.min())

                X[batch_count,:,:,c] = lr_image_array
                Y[batch_count,:,:,c] = sr_image_array

            batch_count = batch_count + 1
            if batch_count >= batch_size:
                break


        yield X, Y, [None]









