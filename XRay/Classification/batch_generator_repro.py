import numpy as np
import random
import ants
import glob

def batch_generator(batch_size,
                    demo):

    # use imagenet mean,std for normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    image_size = (224, 224)
    number_of_channels = 3

    disease_categories = ['Atelectasis',
                          'Cardiomegaly',
                          'Effusion',
                          'Infiltration',
                          'Mass',
                          'Nodule',
                          'Pneumonia',
                          'Pneumothorax',
                          'Consolidation',
                          'Edema',
                          'Emphysema',
                          'Fibrosis',
                          'Pleural_Thickening',
                          'Hernia']
    number_of_dx = len(disease_categories)

    while True:

        X = np.zeros((batch_size, *image_size, number_of_channels))
        Y  = np.zeros((batch_size, number_of_dx))
        
        batch_count = 0
        while batch_count < batch_size:
            random_index = int(random.sample(list(range(demo.shape[0])), 1)[0])
            subject_row = demo.iloc[[random_index]]
            base_image_file = subject_row.index.values[0]
            base_image_file = base_image_file.replace(".png", ".nii.gz")
            image_file = glob.glob("/home/ntustison/Data/reproduce-chexnet/data/nifti/" + base_image_file)
            if len(image_file) > 0:
                image_file = image_file[0]
                image = ants.image_read(image_file)
                if image.components > 1:
                    image_channels = ants.split_channels(image)
                    image = (image_channels[0] + image_channels[1] + image_channels[2]) / 3
                image = ants.resample_image(image, image_size, use_voxels=True, interp_type=0)  
                if len(image.shape) == 2 and image.components == 1 and image.shape == image_size: 
                    image_array = image.numpy()
                    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
                    for c in range(number_of_channels):
                        X[batch_count,:,:,c] = (image_array - imagenet_mean[c]) / (imagenet_std[c])
                    for d in range(number_of_dx):    
                        Y[batch_count,d] = subject_row[disease_categories[d]].values[0]
                    batch_count += 1

        yield X, Y, None

