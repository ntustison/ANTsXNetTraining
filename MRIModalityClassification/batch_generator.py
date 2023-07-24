import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size,
                    image_files,
                    template):

    image_size = template.shape

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
        Y = np.zeros((batch_size, number_of_classes))

        for i in range(batch_size):
            random_modality = random.sample(list(range(number_of_classes)), 1)[0]
            random_index = random.sample(list(range(len(image_files[random_modality]))), 1)[0]       
                
            image = ants.image_read(image_files[random_modality][random_index])
        
            center_of_mass_template = ants.get_center_of_mass(template*0 + 1)
            center_of_mass_image = ants.get_center_of_mass(image*0 + 1)
            translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
            xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
                center=np.asarray(center_of_mass_template), translation=translation)
            image = ants.apply_ants_transform_to_image(xfrm, image, template)

            if random.uniform(0.0, 1.0) < 0.9:
                noise_model = None
                if random.uniform(0.0, 1.0) < 0.33:
                    noise_model = ("additivegaussian", "shot", "saltandpepper")
                data_aug = antspynet.data_augmentation(input_image_list=[[image]],
                                                    segmentation_image_list=None,
                                                    pointset_list=None,
                                                    number_of_simulations=1,
                                                    reference_image=template,
                                                    transform_type='affineAndDeformation',
                                                    noise_model=noise_model,
                                                    sd_simulated_bias_field=1.0,
                                                    sd_histogram_warping=0.05,
                                                    sd_affine=0.05,
                                                    output_numpy_file_prefix=None,
                                                    verbose=False)
                image = data_aug['simulated_images'][0][0]
                
            image = (image - image.min()) / (image.max() - image.min())
            X[i,:,:,:,0] = image.numpy()
            Y[i, random_modality] = 1.0

        yield X, Y, None

