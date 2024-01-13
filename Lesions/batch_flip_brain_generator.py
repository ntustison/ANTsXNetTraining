import numpy as np
import random
import ants
import antspynet
import os

import time

def batch_generator(batch_size=32,
                    template=None,
                    template_mask=None,
                    t1_files=None,
                    lesion_mask_files=None,
                    brain_mask_files=None,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_random_transformation=False):

    while True:

        X = np.zeros((batch_size, *template.shape, 2))
        Y = np.zeros((batch_size, *template.shape, 1))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(t1_files))), 1)[0]

            lesion_mask = ants.threshold_image(ants.image_read(lesion_mask_files[i]), 0, 0, 0, 1)
            brain_mask = ants.threshold_image(ants.image_read(brain_mask_files[i]), 0, 0, 0, 1)
            image = ants.image_read(t1_files[i]) * brain_mask

            if lesion_mask.sum() < 1000:
                continue

            if do_histogram_intensity_warping: # and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                break_points = [0.2, 0.4, 0.6, 0.8]                
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.05)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                image = antspynet.histogram_warp_image_intensities(image,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)
                toc = time.perf_counter()
                # print(f"Histogram warping {toc - tic:0.4f} seconds")

            if do_add_noise and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                image = (image - image.min()) / (image.max() - image.min())
                noise_parameters = (0.0, random.uniform(0, 0.01))
                image = ants.add_noise_to_image(image, noise_model="additivegaussian", noise_parameters=noise_parameters)
                toc = time.perf_counter()
                # print(f"Add noise {toc - tic:0.4f} seconds")

            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                tic = time.perf_counter()
                log_field = antspynet.simulate_bias_field(image, number_of_points=10, sd_bias_field=1.0, number_of_fitting_levels=2, mesh_size=10)
                log_field = log_field.iMath("Normalize")
                field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3, 4), 1)[0])
                image = image * ants.from_numpy(field_array, origin=image.origin, spacing=image.spacing, direction=image.direction)
                image = (image - image.min()) / (image.max() - image.min())
                toc = time.perf_counter()
                # print(f"Sim bias field {toc - tic:0.4f} seconds")

            center_of_mass_template = ants.get_center_of_mass(template_mask)
            center_of_mass_image = ants.get_center_of_mass(brain_mask)
            translation = tuple(np.array(center_of_mass_image) - np.array(center_of_mass_template))
            xfrm = ants.create_ants_transform(transform_type=
                "Euler3DTransform", center = center_of_mass_template,
                translation=translation,
                precision='float', dimension=image.dimension)

            image = ants.apply_ants_transform_to_image(xfrm, image, template)
            brain_mask = ants.apply_ants_transform_to_image(xfrm, brain_mask, template, interpolation="nearestneighbor")
            lesion_mask = ants.apply_ants_transform_to_image(xfrm, lesion_mask, template, interpolation="nearestneighbor")

            if do_random_transformation:
                tic = time.perf_counter()
                data_augmentation = antspynet.randomly_transform_image_data(template,
                    [[image]],
                    [brain_mask],
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
                lesion_mask = ants.apply_ants_transform_to_image(data_augmentation['simulated_transforms'][0], lesion_mask,
                    reference=template, interpolation='nearestneighbor')
                image = data_augmentation['simulated_images'][0][0]
                brain_mask = data_augmentation['simulated_segmentation_images'][0]
                toc = time.perf_counter()

            image_min = image[brain_mask > 0.5].min()
            image_max = image[brain_mask > 0.5].max()
            image = (image - image_min) / (image_max - image_min)

            X[batch_count,:,:,:,0] = image.numpy()
            X[batch_count,:,:,:,1] = np.flip(X[batch_count,:,:,:,0], axis=0)
            Y[batch_count,:,:,:,0] = lesion_mask.numpy()
            
            batch_count = batch_count + 1
            if batch_count >= batch_size:
                break
             
        yield X, Y, None






