import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    image_size=(64, 64),
                    images=None,
                    segmentations=None,
                    labels=None,
                    number_of_slices_per_image=5,
                    do_random_contralateral_flips=True,
                    do_histogram_intensity_warping=True,
                    do_add_noise=True,
                    do_data_augmentation=True):

    if images is None:
        raise ValueError("Input images must be specified.")

    if segmentations is None:
        raise ValueError("Input masks must be specified.")

    while True:

        X = np.zeros((batch_size, *image_size, 2))
        Y = np.zeros((batch_size, *image_size))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(range(len(images)), 1)[0]

            batch_image = ants.image_read(images[i])
            batch_segmentation = ants.image_read(segmentations[i])

            if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
                imageA = batch_image.numpy()
                image = ants.from_numpy(imageA[imageA.shape[0]-1::-1,:,:],
                origin=batch_image.origin, spacing=batch_image.spacing,
                direction=batch_image.direction)
                segmentationA = batch_segmentation.numpy()
                segmentation = ants.from_numpy(segmentationA[segmentationA.shape[0]-1::-1,:,:],
                origin=batch_segmentation.origin, spacing=batch_segmentation.spacing,
                direction=batch_segmentation.direction)
            else:
                image = batch_image
                segmentation = batch_segmentation

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


            imageX = image
            segmentationX = segmentation

            if do_data_augmentation == True:
                data_augmentation = antspynet.randomly_transform_image_data(template,
                    [[image]],
                    [segmentation],
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

                imageX = data_augmentation['simulated_images'][0][0]
                segmentationX = data_augmentation['segmentation_images'][0]

            imageX = (imageX - imageX.mean()) / imageX.std()
            if do_add_noise and random.sample((True, False), 1)[0]:
                noise_parameters = (0.0, random.uniform(0, 0.05))
                imageX = ants.add_noise_to_image(imageX, noise_model="additivegaussian", noise_parameters=noise_parameters)
                imageX = (imageX - imageX.mean()) / imageX.std()

            segmentation_arrayX = segmentationX.numpy()
            image_arrayX = imageX.numpy()

            maskX = ants.threshold_image(segmentationX, 0, 0, 0, 1)
            geoms = ants.label_geometry_measures(maskX)
            if len(geoms['Label']) == 0:
                continue

            image_spacing = ants.get_spacing(imageX)
            which_dimension_max_spacing = image_spacing.index(max(image_spacing))
            if which_dimension_max_spacing == 0:
                lower_slice = geoms['BoundingBoxLower_x'][0]
                upper_slice = geoms['BoundingBoxUpper_x'][0]
            elif which_dimension_max_spacing == 1:
                lower_slice = geoms['BoundingBoxLower_y'][0]
                upper_slice = geoms['BoundingBoxUpper_y'][0]
            else:
                lower_slice = geoms['BoundingBoxLower_z'][0]
                upper_slice = geoms['BoundingBoxUpper_z'][0]
            if lower_slice >= upper_slice:
                continue

            which_random_slices = random.sample(list(range(lower_slice, upper_slice)),
                min(number_of_slices_per_image, upper_slice - lower_slice + 1))
            for j in range(len(which_random_slices)):
                which_slice = which_random_slices[j]

                image_slice = None
                segmentation_slice = None

                if which_dimension_max_spacing == 0:
                    image_slice = ants.from_numpy(np.squeeze(image_arrayX[which_slice,:,:]))
                    segmentation_slice = ants.from_numpy(np.squeeze(segmentation_arrayX[which_slice,:,:]))
                elif which_dimension_max_spacing == 1:
                    image_slice = ants.from_numpy(np.squeeze(image_arrayX[:,which_slice,:]))
                    segmentation_slice = ants.from_numpy(np.squeeze(segmentation_arrayX[:,which_slice,:]))
                else:
                    image_slice = ants.from_numpy(np.squeeze(image_arrayX[:,:,which_slice]))
                    segmentation_slice = ants.from_numpy(np.squeeze(segmentation_arrayX[:,:,which_slice]))

                image_slice = antspynet.pad_or_crop_image_to_size(image_slice, image_size)
                segmentation_slice = antspynet.pad_or_crop_image_to_size(segmentation_slice, image_size)
                mask_slice = ants.smooth_image(ants.threshold_image(segmentation_slice, 0, 0, 0, 1), 1.0)

                X[batch_count,:,:,0] = image_slice.numpy()
                X[batch_count,:,:,1] = mask_slice.numpy()
                Y[batch_count,:,:] = segmentation_slice.numpy()

                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break

        encoded_Y = antspynet.encode_unet(Y, labels)

        yield X, encoded_Y, [None]









