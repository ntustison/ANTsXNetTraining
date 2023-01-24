import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    t1s=None,
                    t1_template=None,
                    t2s=None,
                    t2_template=None,
                    segmentation_images=None,
                    which_hemi="left",
                    priors=None,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True,
                    do_data_augmentation=True):

    if t1s is None or t2s is None or segmentation_images is None:
        raise ValueError("Input images must be specified.")
    if t1_template is None or t2_template is None:
        raise ValueError("Input template must be specified.")

    image_size = (64, 64, 96)

    bounding_box_left = (76, 140, 74, 138, 56, 152)
    bounding_box_right = (20,  84, 74, 138, 56, 152)

    bounding_box = None
    number_of_channels = 2 + len(priors)
    if which_hemi == "left":
        bounding_box = bounding_box_left
    elif which_hemi == "right":
        bounding_box = bounding_box_right
    
    t1_template_roi = ants.crop_indices(t1_template, 
                                        (bounding_box[0], bounding_box[2], bounding_box[4]), 
                                        (bounding_box[1], bounding_box[3], bounding_box[5]))
    t1_template_roi = (t1_template_roi - t1_template_roi.min()) / (t1_template_roi.max() - t1_template_roi.min()) * 2 - 1.0

    t2_template_roi = ants.crop_indices(t2_template, 
                                        (bounding_box[0], bounding_box[2], bounding_box[4]), 
                                        (bounding_box[1], bounding_box[3], bounding_box[5]))
    t2_template_roi = (t2_template_roi - t2_template_roi.min()) / (t2_template_roi.max() - t2_template_roi.min()) * 2 - 1.0

    while True:

        X = np.zeros((batch_size, *image_size, number_of_channels))
        for b in range(batch_size):
            for p in range(len(priors)):
                X[b,:,:,:,p+2] = priors[p][bounding_box[0]:bounding_box[1],
                                           bounding_box[2]:bounding_box[3],
                                           bounding_box[4]:bounding_box[5]]

        Y  = np.zeros((batch_size, *image_size))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(segmentation_images))), 1)[0]

            t1 = ants.image_read(t1s[i])
            t2 = ants.image_read(t2s[i])
            seg = ants.image_read(segmentation_images[i])

            if do_histogram_intensity_warping and random.sample((True, False), 1)[0]:
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.05)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                t1 = antspynet.histogram_warp_image_intensities(t1,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)
                t2 = antspynet.histogram_warp_image_intensities(t2,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)

            if do_add_noise and random.sample((True, False), 1)[0]:
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())
                noise_parameters = (0.0, random.uniform(0, 0.05))
                t1 = ants.add_noise_to_image(t1, noise_model="additivegaussian", noise_parameters=noise_parameters)
                t2 = (t2 - t2.min()) / (t2.max() - t2.min())
                noise_parameters = (0.0, random.uniform(0, 0.05))
                t2 = ants.add_noise_to_image(t2, noise_model="additivegaussian", noise_parameters=noise_parameters)

            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                t1_field = antspynet.simulate_bias_field(t1, sd_bias_field=0.05) 
                t1 = ants.iMath(t1, "Normalize") * (t1_field + 1)
                t1 = (t1 - t1.min()) / (t1.max() - t1.min())
                t2_field = antspynet.simulate_bias_field(t2, sd_bias_field=0.05)
                t2 = ants.iMath(t2, "Normalize") * (t2_field + 1)
                t2 = (t2 - t2.min()) / (t2.max() - t2.min())

            # t1_pre = antspynet.preprocess_brain_image(t1, truncate_intensity=(0.01, 0.995), do_bias_correction=False, do_denoising=False, verbose=False)
            # t1 = t1_pre['preprocessed_image']
            # t2_pre = antspynet.preprocess_brain_image(t2, truncate_intensity=(0.01, 0.995), do_bias_correction=False, do_denoising=False, verbose=False)
            # t2 = t2_pre['preprocessed_image']
            
            if do_data_augmentation == True:
                data_augmentation = antspynet.randomly_transform_image_data(t1,
                    [[t1, t2]],
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
                t1 = data_augmentation['simulated_images'][0][0]
                t2 = data_augmentation['simulated_images'][0][1]
                seg = data_augmentation['simulated_segmentation_images'][0]

            t1_roi = ants.crop_indices(t1, 
                                       (bounding_box[0], bounding_box[2], bounding_box[4]), 
                                       (bounding_box[1], bounding_box[3], bounding_box[5]))              
            t1_roi = ants.histogram_match_image(t1_roi, t1_template_roi, 255, 64, False)

            t2_roi = ants.crop_indices(t2, 
                                       (bounding_box[0], bounding_box[2], bounding_box[4]), 
                                       (bounding_box[1], bounding_box[3], bounding_box[5]))              
            t2_roi = ants.histogram_match_image(t2_roi, t2_template_roi, 255, 64, False)

            seg_roi = ants.crop_indices(seg, 
                                        (bounding_box[0], bounding_box[2], bounding_box[4]), 
                                        (bounding_box[1], bounding_box[3], bounding_box[5]))              

            X[batch_count,:,:,:,0] = t1_roi.numpy()
            X[batch_count,:,:,:,1] = t2_roi.numpy()
            Y[batch_count,:,:,:] = seg_roi.numpy()

            batch_count = batch_count + 1
            if batch_count >= batch_size:
                break

        left_labels = np.array((0, 5, 7, 9, 11, 13, 15, 17))
        right_labels = np.array((0, 6, 8, 10, 12, 14, 16, 18))

        encY = None
        if which_hemi == "left":
            encY = antspynet.encode_unet(Y.astype('int'), left_labels)
        elif which_hemi == "right":
            encY = antspynet.encode_unet(Y.astype('int'), right_labels)

        Y2 = np.sum(encY[:,:,:,:,1:], axis=4)
        Y3 = np.sum(encY[:,:,:,:,1:5], axis=4)
        Y4 = np.sum(encY[:,:,:,:,5:], axis=4)
        yield X, [encY, Y2, Y3, Y4], [None, None, None, None]









