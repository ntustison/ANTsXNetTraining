import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    images=None,
                    label_images=None,
                    left_prior=None,
                    right_prior=None,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_data_augmentation=True):

    if images is None or label_images is None or left_prior is None or right_prior is None:
        raise ValueError("Input images must be specified.")

    image_size = (256, 256)
    number_of_channels = 3
    
    while True:

        X = np.zeros((batch_size, *image_size, number_of_channels))
        Y = np.zeros((batch_size, *image_size))

        for i in range(batch_size):
            X[i,:,:,1] = left_prior.numpy()
            X[i,:,:,2] = right_prior.numpy()

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(images))), 1)[0]
            image = ants.image_read(images[i], dimension=2)
            ants.set_spacing(image, (1, 1))
            labels = ants.image_read(label_images[i], dimension=2)
            ants.set_spacing(labels, (1, 1))
 
            # image = ants.resample_image(image, image_size, use_voxels=True)
            # labels = ants.resample_image(labels, image_size, use_voxels=True)

            if do_histogram_intensity_warping: # and random.sample((True, False), 1)[0]:
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.05)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                image = antspynet.histogram_warp_image_intensities(image,
                    break_points=break_points, clamp_end_points=(True, True),
                    displacements=displacements)

            if do_add_noise: # and random.sample((True, False), 1)[0]:
                image = (image - image.min()) / (image.max() - image.min())
                noise_parameters = (0.0, random.uniform(0, 0.05))
                image = ants.add_noise_to_image(image, noise_model="additivegaussian", noise_parameters=noise_parameters)

            if do_simulate_bias_field: #  and random.sample((True, False), 1)[0]:
                # image_field = antspynet.simulate_bias_field(image, sd_bias_field=0.05) 
                # image = ants.iMath(image, "Normalize") * (image_field + 1)
                # image = (image - image.min()) / (image.max() - image.min())
                log_field = antspynet.simulate_bias_field(image, number_of_points=10, sd_bias_field=1.0, number_of_fitting_levels=2, mesh_size=10)
                log_field = log_field.iMath("Normalize")
                field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3, 4), 1)[0])
                image = image * ants.from_numpy(field_array, origin=image.origin, spacing=image.spacing, direction=image.direction)
                image = (image - image.min()) / (image.max() - image.min())

            if do_data_augmentation:
                sd_noise = 2.0
                data_augmentation = antspynet.randomly_transform_image_data(image,
                    [[image]],
                    [labels],
                    number_of_simulations=1,
                    transform_type='deformation',
                    sd_affine=0.01,
                    deformation_transform_type="bspline",
                    number_of_random_points=100,
                    sd_noise=sd_noise,
                    number_of_fitting_levels=4,
                    mesh_size=1,
                    sd_smoothing=4.0,
                    input_image_interpolator='linear',
                    segmentation_image_interpolator='nearestNeighbor')
                image = data_augmentation['simulated_images'][0][0]
                labels = data_augmentation['simulated_segmentation_images'][0]
            
            image = (image - image.min()) / (image.max() - image.min())  

            X[batch_count,:,:,0] = image.numpy()
            Y[batch_count,:,:] = labels.numpy()

            batch_count = batch_count + 1
            if batch_count >= batch_size:
                break

        encY = antspynet.encode_unet(Y.astype('int'), (0, 1, 2))            
        yield X, encY, None









