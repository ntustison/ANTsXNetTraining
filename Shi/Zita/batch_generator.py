import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    image_size=(64, 64),
                    images=None,
                    segmentation_images=None,
                    segmentation_labels=None,
                    do_random_contralateral_flips=True,
                    do_histogram_intensity_warping=True,
                    do_simulate_bias_field=True,
                    do_add_noise=True):

    if images is None or segmentation_images is None:
        raise ValueError("Input images must be specified.")

    if segmentation_labels is None:
        raise ValueError("segmentation labels must be specified.")

    while True:

        X = np.zeros((batch_size, *image_size, 1))
        Y  = np.zeros((batch_size, *image_size, 1))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(list(range(len(segmentation_images))), 1)[0]

            image = ants.image_read(images[i])
            image = image / image.max()

            seg = ants.image_read(segmentation_images[i])
            seg = ants.copy_image_info(image, seg)

            geoms = ants.label_geometry_measures(seg)
            if len(geoms['Label']) == 0:
                continue

            if do_histogram_intensity_warping and random.sample((True, False), 1)[0]:
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.1)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                image = antspynet.histogram_warp_image_intensities(image,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)

            if do_simulate_bias_field and random.sample((True, False), 1)[0]:
                bias_field = antspynet.simulate_bias_field(image, sd_bias_field=0.02)
                image = ants.iMath(image, "Normalize") * (bias_field + 1)
                image = (image - image.min()) / (image.max() - image.min())

            if do_add_noise and random.sample((True, False), 1)[0]:
                image = (image - image.min()) / (image.max() - image.min())
                noise_parameters = (0.0, random.uniform(0, 0.05))
                image = ants.add_noise_to_image(image, noise_model="additivegaussian", noise_parameters=noise_parameters)

            image_array = (image.numpy()).astype('float64')
            seg_array = seg.numpy()

            if do_random_contralateral_flips:
                if random.sample((True, False), 1)[0]:
                    image_array = np.fliplr(image_array)
                    seg_array = np.fliplr(seg_array)
                if random.sample((True, False), 1)[0]:
                    image_array = np.flipud(image_array)
                    seg_array = np.flipud(seg_array)
                if random.sample((True, False), 1)[0]:
                    number_of_rot90s = random.sample((1, 2, 3), 1)[0]
                    image_array = np.rot90(image_array, k=number_of_rot90s)
                    seg_array = np.rot90(seg_array, k=number_of_rot90s)

            image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            seg_array[seg_array != 0] = 1

            X[batch_count,:,:,0] = image_array
            Y[batch_count,:,:,0] = seg_array

            batch_count = batch_count + 1
            if batch_count >= batch_size:
                break

            if batch_count >= batch_size:
                break

        # ants.image_write(ants.from_numpy(np.squeeze(X)), "X.nii.gz")
        # ants.image_write(ants.from_numpy(np.squeeze(Y)), "Y.nii.gz")
        # raise ValueError("Done")

        # encY = antspynet.encode_unet(Y, segmentation_labels)

        yield X, Y, [None]









