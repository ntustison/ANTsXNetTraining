import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    patch_size=(64, 64, 64),
                    template=None,
                    priors=None,
                    images=None,
                    segmentation_images=None,
                    segmentation_labels=None,
                    do_random_contralateral_flips=True,
                    do_histogram_intensity_warping=True,
                    do_add_noise=True,
                    do_data_augmentation=True):

    if template is None or priors is None:
        raise ValueError("Template/priors must be specified.")

    if images is None or segmentation_images is None:
        raise ValueError("Input images must be specified.")

    if segmentation_labels is None:
        raise ValueError("segmentation labels must be specified.")

    stride_length = tuple(np.array(template.shape) - np.array(patch_size))

    priors_patches = list()
    for i in range(len(priors)):
        priors_patches.append( 
            antspynet.extract_image_patches(priors[i], patch_size, max_number_of_patches="all",
                stride_length=stride_length, random_seed=None, return_as_array=True)
            )
        

    while True:

        X = np.zeros((batch_size, *patch_size, 2))
        Y  = np.zeros((batch_size, *patch_size))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(segmentation_images))), 1)[0]

            t1 = ants.image_read(images[i])
            seg = ants.image_read(segmentation_images[i])

            seg = ants.copy_image_info(t1, seg)

            if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
                t1 = ants.reflect_image(t1, axis=0)
                seg = ants.reflect_image(seg, axis=0)

            if do_histogram_intensity_warping and random.sample((True, False), 1)[0]:
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.175)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                t1 = antspynet.histogram_warp_image_intensities(t1,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)

            if do_data_augmentation == True:
                data_augmentation = antspynet.randomly_transform_image_data(t1,
                    [[t1]],
                    [seg],
                    number_of_simulations=1,
                    transform_type='affine',
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
                seg = data_augmentation['segmentation_images'][0]

            t1 = (t1 - t1.min()) / (t1.max() - t1.min())
            if do_add_noise and random.sample((True, False), 1)[0]:
                noise_parameters = (0.0, random.uniform(0, 0.05))
                t1 = ants.add_noise_to_image(t1, noise_model="additivegaussian", noise_parameters=noise_parameters)

            t1 = (t1 - t1.mean()) / t1.std() 

            image_patches = antspynet.extract_image_patches(t1, patch_size, max_number_of_patches="all",
                stride_length=stride_length, random_seed=None, return_as_array=True)
            seg_patches = antspynet.extract_image_patches(seg, patch_size, max_number_of_patches="all",
                stride_length=stride_length, random_seed=None, return_as_array=True)


            X[batch_count,:,:,:,0] = image_patches[batch_count%8,:,:,:]
            X[batch_count,:,:,:,1] = priors_patches[6][batch_count%8,:,:,:]
#            for p in range(len(priors)):
#                X[batch_count,:,:,:,1+p] = priors_patches[p][batch_count%8,:,:,:]
            Y[batch_count,:,:,:] = seg_patches[batch_count%8,:,:,:]

            batch_count = batch_count + 1
            if batch_count >= batch_size:
                break

        encoded_Y = antspynet.encode_unet(Y, segmentation_labels)

        yield X, encoded_Y, [None]









