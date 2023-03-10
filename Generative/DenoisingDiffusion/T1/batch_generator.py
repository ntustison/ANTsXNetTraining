import numpy as np
import random
import ants
import antspynet
import cv2


def batch_generator(batch_size=32,
                    t1s=None,
                    image_size=None,
                    number_of_channels=1,
                    template=None,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_data_augmentation=True):

    verbose = False

    template_lower = 58 + 10
    template_upper = 188 - 10
    slices_per_subject = 4

    while True:

        X = np.zeros((batch_size, *image_size, number_of_channels))
        Y = np.zeros((batch_size, *image_size, number_of_channels))

        batch_count = 0

        while batch_count < batch_size:

            if verbose:
                print("Batch count: " + str(batch_count))

            if verbose:
                print("    Augment input T1")

            # Augment the input T1

            which_t1 = random.sample(list(range(len(t1s))), 1)[0]

            t1 = ants.image_read(t1s[which_t1])

            center_of_mass_template = ants.get_center_of_mass(template * 0 + 1)
            center_of_mass_image = ants.get_center_of_mass(t1 * 0 + 1)
            translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
            xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
                center=np.asarray(center_of_mass_template), translation=translation)
            t1 = xfrm.apply_to_image(t1, template)

            if do_data_augmentation == True:
                data_augmentation = antspynet.randomly_transform_image_data(template,
                    [[t1]],
                    [brain_mask],
                    number_of_simulations=1,
                    transform_type='deformation',
                    sd_affine=0.01,
                    deformation_transform_type="bspline",
                    number_of_random_points=1000,
                    sd_noise=2.0,
                    number_of_fitting_levels=4,
                    mesh_size=1,
                    sd_smoothing=4.0,
                    input_image_interpolator='linear',
                    segmentation_image_interpolator='nearestNeighbor')
                t1 = data_augmentation['simulated_images'][0][0]
                brain_mask = data_augmentation['simulated_segmentation_images'][0]

            if do_histogram_intensity_warping and random.sample((True, False), 1)[0]:
                if verbose:
                    print("    Histogram intensity warping.")
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.05)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                t1 = antspynet.histogram_warp_image_intensities(t1,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)

            if do_add_noise and random.sample((True, False), 1)[0]:
                if verbose:
                    print("    Add noise.")
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())
                noise_parameters = (0.0, random.uniform(0, 0.05))
                t1 = ants.add_noise_to_image(t1, noise_model="additivegaussian", noise_parameters=noise_parameters)

            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                if verbose:
                    print("    Simulate bias field.")

                log_field = antspynet.simulate_bias_field(t1, number_of_points=10, sd_bias_field=1.0,
                             number_of_fitting_levels=2, mesh_size=10)
                log_field = log_field.iMath("Normalize")
                field_array = np.power(np.exp(log_field.numpy()), random.sample([3,4,5], 1)[0])
                t1 = t1 * ants.from_numpy(field_array, origin=t1.origin,
                               spacing=t1.spacing, direction=t1.direction)

            quantiles = (t1.quantile(0.01), t1.quantile(0.99))
            t1[t1 < quantiles[0]] = quantiles[0]
            t1[t1 > quantiles[1]] = quantiles[1]

            slice_numbers = random.sample(list(range(template_lower, template_upper)), slices_per_subject)
            for i in range(len(slice_numbers)):
                slice = ants.slice_image(t1, axis=1, idx=slice_numbers[i], collapse_strategy=1)
                slice = antspynet.pad_or_crop_image_to_size(slice, image_size)
                slice = (slice - slice.min()) / (slice.max() - slice.min()) * 2 - 1

                for j in range(number_of_channels):
                    X[batch_count,:,:,j] = slice.numpy()
                    Y[batch_count,:,:,j] = slice.numpy()

                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break

            if batch_count >= batch_size:
                break

        yield X, Y

