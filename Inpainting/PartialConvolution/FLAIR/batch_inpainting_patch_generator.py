import numpy as np
import random
import ants
import antspynet
import skimage

def batch_generator(batch_size=32,
                    patch_size=(64, 64, 64),
                    t1s=None,
                    number_of_channels=1,
                    template=None,
                    template_labels=None,
                    template_roi=None,
                    template_priors=None,
                    add_2d_masking=False,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_data_augmentation=True,
                    return_ones_masks=False,
                    verbose=False):


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
        mask = ants.threshold_image(field_image, -1.75, 1.75, 1, 0)
        return mask

    def create_random_mask2_3d(template,
                               template_labels,
                               roi=None):

        labels_image = template_labels
        labels_array = labels_image.numpy()

        unique_labels = labels_image.unique().astype(int)[1:]
        random_labels = random.sample(list(unique_labels), random.randint(2,10))
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

    def create_random_lines_3d(domain_image,
                               max_number_of_lines=50,
                               max_dilation_radius=10):
        image = ants.image_clone(domain_image) * 0

        # Draw random lines
        for _ in range(random.randint(1, max_number_of_lines)):
            x1 = random.randint(1, domain_image.shape[0]-2)
            x2 = random.randint(1, domain_image.shape[0]-2)
            y1 = random.randint(1, domain_image.shape[1]-2)
            y2 = random.randint(1, domain_image.shape[1]-2)
            z1 = random.randint(1, domain_image.shape[2]-2)
            z2 = random.randint(1, domain_image.shape[2]-2)
            line_indices = skimage.draw.line_nd((x1, y1, z1), (x2, y2, z2), endpoint=True)
            single_line_image_array = np.zeros(domain_image.shape)
            single_line_image_array[line_indices] = 1
            single_line_image = ants.from_numpy(np.squeeze(single_line_image_array),
                origin=domain_image.origin, spacing=domain_image.spacing,
                direction=domain_image.direction)
            single_line_image = ants.iMath_MD(single_line_image,
                radius=random.randint(4, max_dilation_radius))
            image = image + single_line_image

        image = ants.threshold_image(image, 0, 0, 0, 1)

        mask = image * -1.0 + 1
        return mask


    stride_length = np.array(patch_size) // 2
    max_number_of_patches_per_subject = 5

    while True:

        X = np.zeros((batch_size, *patch_size, number_of_channels))
        XMask = np.ones((batch_size, *patch_size, number_of_channels))
        Y = np.zeros((batch_size, *patch_size, number_of_channels))

        batch_count = 0

        while batch_count < batch_size:

            if verbose:
                print("Batch count: " + str(batch_count))


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

                if verbose:
                    print("    Augment input T1")
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
            # t1[t1 < quantiles[0]] = quantiles[0]
            # t1[t1 > quantiles[1]] = quantiles[1]

            if not return_ones_masks:
                # mask = create_random_mask_3d(t1)
                # mask_inverted = (mask * -1.0 + 1.0) # * template_roi
                # mask = mask_inverted * -1.0 + 1.0

                # mask2 = create_random_mask2_3d(template, template_labels, template_roi)
                # mask = mask * mask2

                mask = create_random_lines_3d(template)
                # mask = mask * mask3

            else:
                mask = ants.image_clone(t1) * 0 + 1

            t1 = (t1 - t1[mask == 1].min()) / (t1[mask == 1].max() - t1[mask == 1].min())

            image_patches = antspynet.extract_image_patches(t1, patch_size, max_number_of_patches="all",
                stride_length=stride_length, random_seed=None, return_as_array=True)
            mask_patches = antspynet.extract_image_patches(mask, patch_size, max_number_of_patches="all",
                stride_length=stride_length, random_seed=None, return_as_array=True)

            number_of_patches = image_patches.shape[0]

            sample_patch_indices = random.sample(range(number_of_patches), max_number_of_patches_per_subject)
            for i in range(len(sample_patch_indices)):

                image_patch = image_patches[sample_patch_indices[i],:,:,:]
                mask_patch = mask_patches[sample_patch_indices[i],:,:,:]
                masked_image_patch = image_patch * mask_patch
                masked_image_patch[mask_patch == 0] = 1

                X[batch_count,:,:,:,0] = masked_image_patch
                XMask[batch_count,:,:,:,0] = mask_patch
                Y[batch_count,:,:,:,0] = image_patch

                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break

            if batch_count >= batch_size:
                break

        yield [X, XMask], Y

