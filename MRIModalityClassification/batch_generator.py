import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size,
                    image_files,
                    template,
                    modalities):

    image_size = template.shape

    if len(image_files) != len(modalities):
        raise ValueError("Image files must match the modalities")

    # modalities:
    #     T1
    #     T2
    #     FLAIR
    #     T2Star
    #     Mean DTI
    #     Mean Bold
    #     ASL
    number_of_classes = 7

    while True:

        X = np.zeros((batch_size, *image_size, 1))
        Y  = np.zeros((batch_size, number_of_classes))

        random_indices = random.sample(list(range(len(image_files))), batch_size)
        for i in range(len(random_indices)):
            image = ants.image_read(image_files[random_indices[i]])

            # if random.uniform(0, 1.0) < 0.75:
            data_aug = antspynet.data_augmentation(input_image_list=[[image]],
                                                   segmentation_image_list=None,
                                                   pointset_list=None,
                                                   number_of_simulations=1,
                                                   reference_image=template,
                                                   transform_type='affineAndDeformation',
                                                   noise_model=("additivegaussian", "shot", "saltandpepper"),
                                                   sd_simulated_bias_field=1.0,
                                                   sd_histogram_warping=0.05,
                                                   sd_affine=0.05,
                                                   output_numpy_file_prefix=None,
                                                   verbose=False)
            image = data_aug['simulated_images'][0][0]
            image = (image - image.min()) / (image.max() - image.min())
            X[i,:,:,:,0] = image.numpy()
            Y[i, modalities[random_indices[i]]] = 1.0

        yield X, Y, None

