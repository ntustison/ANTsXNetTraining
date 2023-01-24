import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    segmentation_images=None,
                    priors=None,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_data_augmentation=True,
                    use_two_outputs=False):

    if segmentation_images is None:
        raise ValueError("Input images must be specified.")

    while True:

        image_size = priors[0].shape
        number_of_channels = 1 + len(priors)

        X = np.zeros((batch_size, *image_size, number_of_channels))
        for b in range(batch_size):
            for p in range(len(priors)):
                X[b,:,:,:,p+1] = priors[p]

        Y  = np.zeros((batch_size, *image_size))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(segmentation_images))), 1)[0]

            seg = ants.image_read(segmentation_images[i])
            h1 = ants.threshold_image(seg, 0, 0, 0, 1)

            if do_histogram_intensity_warping: # and random.sample((True, False), 1)[0]:
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.05)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                h1 = antspynet.histogram_warp_image_intensities(h1,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)

            if do_add_noise: # and random.sample((True, False), 1)[0]:
                h1 = (h1 - h1.min()) / (h1.max() - h1.min())
                noise_parameters = (0.0, random.uniform(0, 0.05))
                h1 = ants.add_noise_to_image(h1, noise_model="additivegaussian", noise_parameters=noise_parameters)

            if do_simulate_bias_field: # and random.sample((True, False), 1)[0]:
                h1_field = antspynet.simulate_bias_field(h1, sd_bias_field=0.05) 
                h1 = ants.iMath(h1, "Normalize") * (h1_field + 1)
                h1 = (h1 - h1.min()) / (h1.max() - h1.min())

            # h1_pre = antspynet.preprocess_brain_image(h1, truncate_intensity=(0.01, 0.995), do_bias_correction=False, do_denoising=False, verbose=False)
            # h1 = h1_pre['preprocessed_image']
            # h1 = (h1 - h1.mean()) / h1.std()

            if do_data_augmentation == True:
                data_augmentation = antspynet.randomly_transform_image_data(h1,
                    [[h1]],
                    [seg],
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
                h1 = data_augmentation['simulated_images'][0][0]
                seg = data_augmentation['simulated_segmentation_images'][0]

            # ants.image_write(ants.from_numpy(h1_array), "h1_" + str(batch_count) + ".nii.gz")
            # ants.image_write(ants.from_numpy(seg_array), "seg_" + str(batch_count) + ".nii.gz")
            # print("batch_count = ", str(batch_count))

            X[batch_count,:,:,:,0] = h1.numpy()
            Y[batch_count,:,:,:] = np.round(seg.numpy())

            batch_count = batch_count + 1
            if batch_count >= batch_size:
                break

        labels = np.array((0, 1, 2, 3, 4, 5))

        encY = antspynet.encode_unet(Y.astype('int'), labels)
            
        if use_two_outputs:
            Y2 = np.sum(encY[:,:,:,:,1:], axis=4)
            yield X, [encY, Y2], [None, None]
        else:    
            yield X, encY, [None]









