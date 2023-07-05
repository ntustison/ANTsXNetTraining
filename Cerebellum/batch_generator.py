import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    t1s=None,
                    t1_template=None,
                    labels_images=None,
                    tissue_images=None,
                    priors=None,
                    regional_labels=None,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_data_augmentation=True,
                    which_model=None):

    if t1s is None or labels_images is None or tissue_images is None:
        raise ValueError("Input images must be specified.")
    if t1_template is None:
        raise ValueError("Input template must be specified.")

    image_size = t1_template.shape

    if which_model == "whole":
        number_of_channels = 2 
    elif which_model == "labels_ala_dkt":     
        number_of_channels = 1
    elif which_model == "tissue":
        number_of_channels = 4 
    elif which_model == "labels":
        number_of_channels = 1 + len(regional_labels)
    else:
        number_of_channels = 1 + len(priors)
    
    while True:

        X = np.zeros((batch_size, *image_size, number_of_channels))
        for b in range(batch_size):
            for p in range(1, number_of_channels):
                X[b,:,:,:,p] = priors[p-1].numpy()
        Y  = np.zeros((batch_size, *image_size))
        Y3  = np.zeros((batch_size, *image_size))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(t1s))), 1)[0]

            t1 = antspynet.pad_or_crop_image_to_size(ants.image_read(t1s[i]), image_size)
            labels = antspynet.pad_or_crop_image_to_size(ants.image_read(labels_images[i]), image_size)
            tissue = antspynet.pad_or_crop_image_to_size(ants.image_read(tissue_images[i]), image_size)

            if do_histogram_intensity_warping: # and random.sample((True, False), 1)[0]:
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
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())
                noise_parameters = (0.0, random.uniform(0, 0.05))
                t1 = ants.add_noise_to_image(t1, noise_model="additivegaussian", noise_parameters=noise_parameters)

            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                # t1_field = antspynet.simulate_bias_field(t1, sd_bias_field=0.05) 
                # t1 = ants.iMath(t1, "Normalize") * (t1_field + 1)
                # t1 = (t1 - t1.min()) / (t1.max() - t1.min())
                log_field = antspynet.simulate_bias_field(t1, number_of_points=10, sd_bias_field=1.0, number_of_fitting_levels=2, mesh_size=10)
                log_field = log_field.iMath("Normalize")
                field_array = np.power(np.exp(log_field.numpy()), random.sample((2, 3, 4), 1)[0])
                t1 = t1 * ants.from_numpy(field_array, origin=t1.origin, spacing=t1.spacing, direction=t1.direction)
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())

            if do_data_augmentation == True:
                if which_model == "whole":
                    sd_noise = 5.0
                else:
                    sd_noise = 2.0
                data_augmentation = antspynet.randomly_transform_image_data(t1,
                    [[t1]],
                    [labels],
                    number_of_simulations=1,
                    transform_type='deformation',
                    sd_affine=0.01,
                    deformation_transform_type="bspline",
                    number_of_random_points=1000,
                    sd_noise=sd_noise,
                    number_of_fitting_levels=4,
                    mesh_size=1,
                    sd_smoothing=4.0,
                    input_image_interpolator='linear',
                    segmentation_image_interpolator='nearestNeighbor')
                tissue = ants.apply_ants_transform_to_image(data_augmentation['simulated_transforms'][0], tissue,
                    reference=t1_template, interpolation='nearestneighbor')
                t1 = data_augmentation['simulated_images'][0][0]
                labels = data_augmentation['simulated_segmentation_images'][0]

            if which_model == "tissue" or which_model == "labels" or which_model == "labels_hybrid" or which_model == "labels_ala_dkt":
                t1 *= ants.threshold_image(tissue, 0, 0, 0, 1)
            
            t1 = (t1 - t1.min()) / (t1.max() - t1.min())  

            X[batch_count,:,:,:,0] = t1.numpy()
            Y[batch_count,:,:,:] = labels.numpy()
            Y3[batch_count,:,:,:] = tissue.numpy()

            batch_count = batch_count + 1
            if batch_count >= batch_size:
                break

        encY = antspynet.encode_unet(Y.astype('int'), (0, *regional_labels))
        encY3 = antspynet.encode_unet(Y3.astype('int'), (0, 1, 2, 3))
        Y2 = np.sum(encY3[:,:,:,:,1:], axis=4)
        encY2 = antspynet.encode_unet(Y2.astype('int'), (0, 1))
            
        if which_model == "whole":
            yield X, encY2, None
        elif which_model == "tissue":
            yield X, encY3, None
        elif which_model == "labels" or which_model == "labels_ala_dkt" or which_model == "labels_hybrid":    
            yield X, encY, None
        else:    
            yield X, [encY, Y2, encY3], [None, None, None]









