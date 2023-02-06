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
                    template_labels=None,
                    template_roi=None,
                    template_priors=None,
                    add_2d_masking=False,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_data_augmentation=True):


    def create_random_mask_3d(domain_image,
                              max_number_of_points=100,
                              max_mesh_size=10):
        number_of_points = random.randint(10, max_number_of_points)
        mesh_size = random.randint(5, max_mesh_size)
        log_field = antspynet.simulate_bias_field(domain_image,
                                                 number_of_points=number_of_points,
                                                 mesh_size=mesh_size)
        log_field = log_field.iMath("Normalize")
        field_array = np.power(np.exp(log_field.numpy()), random.randint(3, 5))
        field_image = ants.from_numpy(field_array, origin=domain_image.origin,
               spacing=domain_image.spacing, direction=domain_image.direction)
        field_image = (field_image - field_image.mean()) / field_image.std()
        mask = ants.threshold_image(field_image, -1.35, 1.35, 1, 0)
        return mask

    def create_random_mask_2d(domain_image,
                              max_number_of_lines=5,
                              max_line_thickness=8,
                              max_number_of_ellipses=5):
        image_array = np.zeros((*domain_image.shape, 3))

        # Draw random lines
        for _ in range(random.randint(1, max_number_of_lines)):
            x1 = random.randint(1, domain_image.shape[0])
            x2 = random.randint(1, domain_image.shape[0])
            y1 = random.randint(1, domain_image.shape[1])
            y2 = random.randint(1, domain_image.shape[1])
            thickness = random.randint(1, max_line_thickness)
            cv2.line(image_array, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        # Draw random ellipses
        for _ in range(random.randint(1, max_number_of_ellipses)):
            x1 = random.randint(1, domain_image.shape[0])
            y1 = random.randint(1, domain_image.shape[1])
            s1 = random.randint(1, domain_image.shape[0])
            s2 = random.randint(1, domain_image.shape[1])
            a1 = random.randint(1, 180)
            a2 = random.randint(1, 180)
            a3 = random.randint(1, 180)
            cv2.ellipse(image_array, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), -1)

        image = ants.from_numpy(np.squeeze(image_array[:,:,0]), origin=domain_image.origin,
            spacing=domain_image.spacing, direction=domain_image.direction)

        mask = image * -1.0 + 1
        return mask

    def create_random_mask2_3d(template,
                               template_labels,
                               roi=None):

        labels_image = template_labels
        labels_array = labels_image.numpy()

        unique_labels = labels_image.unique().astype(int)[1:]
        random_labels = random.sample(list(unique_labels), random.randint(2,4))
        random_labels_array = np.zeros_like(labels_array)
        for i in range(len(random_labels)):
            random_labels_array[labels_array == random_labels[i]] = 1

        random_image = ants.from_numpy(random_labels_array, origin=labels_image.origin,
            spacing=labels_image.spacing, direction=labels_image.direction)

        data_augmentation = antspynet.randomly_transform_image_data(template,
            [[template]],
            [random_image],
            number_of_simulations=1,
            transform_type='affineAndDeformation',
            sd_affine=0.1,
            deformation_transform_type="bspline",
            number_of_random_points=1000,
            sd_noise=10.0,
            number_of_fitting_levels=2,
            mesh_size=5,
            sd_smoothing=4.0,
            input_image_interpolator='linear',
            segmentation_image_interpolator='nearestNeighbor')

        mask = data_augmentation['simulated_segmentation_images'][0]
        if roi is not None:
            mask = mask * roi
        mask = mask * -1.0 + 1

        return mask


    verbose = False

    template_lower = 58 + 10
    template_upper = 188 - 10
    slices_per_subject = 4

    while True:

        X = np.zeros((batch_size, *image_size, number_of_channels))
        XMask = np.ones((batch_size, *image_size, number_of_channels))
        Y = np.zeros((batch_size, *image_size, number_of_channels))

        XPriors = None
        if template_priors is not None:
            XPriors = np.zeros((batch_size, *image_size, len(template_priors)))

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

            mask = create_random_mask_3d(t1)
            mask_inverted = (mask * -1.0 + 1.0) * template_roi
            mask = mask_inverted * -1.0 + 1.0

            # mask = create_random_mask2_3d(template, template_labels, template_roi)

            slice_numbers = random.sample(list(range(template_lower, template_upper)), slices_per_subject)
            for i in range(len(slice_numbers)):
                slice = ants.slice_image(t1, axis=1, idx=slice_numbers[i], collapse_strategy=1)
                slice = antspynet.pad_or_crop_image_to_size(slice, image_size)
                slice = (slice - slice.min()) / (slice.max() - slice.min())

                template_slice = ants.slice_image(template, axis=1, idx=slice_numbers[i], collapse_strategy=1)
                template_slice = antspynet.pad_or_crop_image_to_size(template_slice, image_size)
                template_slice = (template_slice - template_slice.min()) / (template_slice.max() - template_slice.min())

                mask_slice = ants.slice_image(mask, axis=1, idx=slice_numbers[i], collapse_strategy=1)
                if add_2d_masking:
                    mask_slice = create_random_mask_2d(mask_slice) * mask_slice
                    if template_roi is not None:
                        template_roi_slice = ants.slice_image(template_roi, axis=1, idx=slice_numbers[i], collapse_strategy=1)
                        mask_slice = ((mask_slice * -1.0 + 1.0) * template_roi_slice) * -1.0 + 1.0
                mask_slice_inverted = ants.threshold_image(mask_slice, 0, 0, 1, 0)
                mask_slice_inverted = antspynet.pad_or_crop_image_to_size(mask_slice_inverted, image_size)
                mask_slice = ants.threshold_image(mask_slice_inverted, 0, 0, 1, 0)
                slice_masked = slice * mask_slice
                slice_masked[mask_slice == 0] = template_slice[mask_slice == 0]

                for j in range(number_of_channels):
                    X[batch_count,:,:,j] = slice_masked.numpy()
                    XMask[batch_count,:,:,j] = mask_slice.numpy()
                    Y[batch_count,:,:,j] = slice.numpy()

                if template_priors is not None:
                    for j in range(len(template_priors)):
                        template_prior_slice = ants.slice_image(template_priors[j], axis=1, idx=slice_numbers[i], collapse_strategy=1)
                        template_prior_slice = antspynet.pad_or_crop_image_to_size(template_prior_slice, image_size)
                        XPriors[batch_count,:,:,j] = template_prior_slice.numpy()


                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break

            if batch_count >= batch_size:
                break

        if template_priors is not None:
            yield [X, XMask, XPriors], Y
        else:
            yield [X, XMask], Y

