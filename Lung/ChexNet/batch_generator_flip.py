import numpy as np
import random
import ants
import glob
import os

def batch_generator(batch_size,
                    image_files,
                    image_size):

    while True:

        X = np.zeros((batch_size, *image_size, 1))
        Y  = np.zeros((batch_size, 3))

        batch_count = 0
        while batch_count < batch_size:
            random_index = random.sample(list(range(len(image_files))), 1)[0]            
            image_file = glob.glob("/home/ntustison/Data/XRayCT/Data/*/" + image_files[random_index])
            if len(image_file) > 0:
                image_file = image_file[0]
                image = ants.image_read(image_file)
                if len(image.shape) == 2 and image.components == 1:       
                    image = ants.resample_image(image, image_size, use_voxels=True)
                    # 1:  Flip upside down
                    # 2:  Flip left right
                    tri_coin_flip = random.sample((0, 1, 2), 1)[0]
                    if tri_coin_flip == 1:
                        image = ants.from_numpy(np.fliplr(image.numpy()), origin=image.origin, spacing=image.spacing, direction=image.direction)
                    elif tri_coin_flip == 2:
                        image = ants.from_numpy(np.flipud(image.numpy()), origin=image.origin, spacing=image.spacing, direction=image.direction)
                    image = (image - image.min()) / (image.max() - image.min())
                    X[batch_count,:,:,0] = image.numpy()
                    # X[batch_count,:,:,1] = X[batch_count,:,:,0]
                    # X[batch_count,:,:,2] = X[batch_count,:,:,0]
                    Y[batch_count, tri_coin_flip] = 1
                    batch_count += 1

        yield X, Y, None

