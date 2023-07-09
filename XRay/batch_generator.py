import numpy as np
import random
import ants
import glob
import os

def batch_generator(batch_size,
                    image_files,
                    demo):

    image_size = (1024, 1024)

    while True:

        X = np.zeros((batch_size, *image_size, 1))
        Y  = np.zeros((batch_size, demo.shape[1]))

        batch_count = 0
        while batch_count < batch_size:
            random_index = random.sample(list(range(len(image_files))), 1)[0]            
            image_file = glob.glob("/home/ntustison/Data/XRayCT/Data/*/" + image_files[random_index])
            if len(image_file) > 0:
                image_file = image_file[0]
                image = ants.image_read(image_file)
                if len(image.shape) == 2 and image.components == 1 and image.shape == image_size:
                    image = (image - image.min()) / (image.max() - image.min())
                    X[batch_count,:,:,0] = image.numpy()
                    Y[batch_count,:] = demo[random_index,:]
                    batch_count += 1

        yield X, Y, None

