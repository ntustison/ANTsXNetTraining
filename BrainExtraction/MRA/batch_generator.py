import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    image_size=(64, 64),
                    template=None,
                    template_brain_mask=None,
                    images=None,
                    brain_masks=None,
                    do_random_contralateral_flips=True,
                    do_histogram_intensity_warping=True,
                    do_add_noise=True,
                    do_simulate_bias_field=True,
                    do_data_augmentation=True):

    if images is None:
        raise ValueError("Input images must be specified.")

    if brain_masks is None:
        raise ValueError("Input masks must be specified.")

    if template is None or template_brain_mask is None:
        raise ValueError("Template must be specified")

    while True:

        X = np.zeros((batch_size, *image_size, 1))
        Y = np.zeros((batch_size, *image_size, 1))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(range(len(images)), 1)[0]

            batch_image = ants.image_read(images[i])
            batch_brain_mask = ants.image_read(brain_masks[i])

            if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
                imageA = batch_image.numpy()
                image = ants.from_numpy(imageA[imageA.shape[0]-1::-1,:,:],
                origin=batch_image.origin, spacing=batch_image.spacing,
                direction=batch_image.direction)
                brain_maskA = batch_brain_mask.numpy()
                brain_mask = ants.from_numpy(brain_maskA[brain_maskA.shape[0]-1::-1,:,:],
                origin=batch_brain_mask.origin, spacing=batch_brain_mask.spacing,
                direction=batch_brain_mask.direction)
            else:
                image = batch_image
                brain_mask = batch_brain_mask

            if do_histogram_intensity_warping and random.sample((True, False), 1)[0]:
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.175)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                image = antspynet.histogram_warp_image_intensities(image,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)

            center_of_mass_template = ants.get_center_of_mass(template)
            center_of_mass_image = ants.get_center_of_mass(image)
            translation = tuple(np.array(center_of_mass_image) - np.array(center_of_mass_template))
            xfrm = ants.create_ants_transform(transform_type=
                "Euler3DTransform", center = center_of_mass_template,
                translation=translation,
                precision='float', dimension=image.dimension)

            imageX = ants.apply_ants_transform_to_image(xfrm, image, template)
            maskX = ants.apply_ants_transform_to_image(xfrm, brain_mask, template, interpolation="nearestneighbor")

            if do_data_augmentation:
                data_augmentation = antspynet.randomly_transform_image_data(template,
                    [[imageX]],
                    [maskX],
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

                imageX = data_augmentation['simulated_images'][0][0]
                maskX = data_augmentation['simulated_segmentation_images'][0]

            imageX = (imageX - imageX.mean()) / imageX.std()
            if do_add_noise and random.sample((True, False), 1)[0]:
                noise_parameters = (0.0, random.uniform(0, 0.05))
                imageX = ants.add_noise_to_image(imageX, noise_model="additivegaussian", noise_parameters=noise_parameters)
                imageX = (imageX - imageX.min()) / (imageX.max() - imageX.min())

            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                log_field = antspynet.simulate_bias_field(imageX, 
                                                          number_of_points=10, 
                                                          sd_bias_field=1.0, 
                                                          number_of_fitting_levels=2, 
                                                          mesh_size=8)
                log_field = log_field.iMath("Normalize")
                field_array = np.power(np.exp(log_field.numpy()), 
                                       random.sample((2, 3, 4), 1)[0])
                imageX = imageX * ants.from_numpy(field_array, 
                                                  origin=imageX.origin, 
                                                  spacing=imageX.spacing, 
                                                  direction=imageX.direction)
                imageX = (imageX - imageX.min()) / (imageX.max() - imageX.min())

            imageX = (imageX - imageX.min()) / (imageX.max() - imageX.min()) 
            X[batch_count,:,:,:,0] = imageX.numpy()
            Y[batch_count,:,:,:,0] = maskX.numpy()

            batch_count = batch_count + 1
            if batch_count >= batch_size:
                break

        # encoded_Y = antspynet.encode_unet(Y, (0, 1))

        yield X, Y









