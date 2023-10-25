import ants
import antspynet
import numpy as np
import random
import glob

import matplotlib.pyplot as plt

base_directory = '/Users/ntustison/Data/Public/XRayCT/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + "Data/"



################################################
#
#  Load the data
#
################################################

test_images_file = base_directory + "CXR8-selected/test_list.txt"
with open(test_images_file) as f:
    test_images_list = f.readlines()
f.close()
test_images_list = [x.strip() for x in test_images_list]

################################################
#
#  Create the model and load weights
#
################################################

image_size = (1024, 1024)
image_size = (224, 224)

model = antspynet.create_resnet_model_2d((None, None, 1),
   number_of_classification_labels=3,
   mode="classification",
   layers=(1, 2, 3, 4),
   residual_block_schedule=(2, 2, 2, 2), lowest_resolution=64,
   cardinality=1, squeeze_and_excite=False)

weights_filename = scripts_directory + "xray_flip_classification.h5"
model.load_weights(weights_filename)

################################################
#
#  Test the model
#
################################################

batchX = np.zeros((1,*image_size,1))    

for i in range(10):
    print("Iteration: " + str(i))
    random_index = random.sample(list(range(len(test_images_list))), 1)[0]
    image_file = glob.glob(data_directory + "/*/" + test_images_list[random_index])
    if len(image_file) > 0:
        image_file = image_file[0]
        image = ants.image_read(image_file)
        image = ants.resample_image(image, image_size, use_voxels=True)
        if len(image.shape) == 2 and image.components == 1:       
            image = ants.resample_image(image, image_size, use_voxels=True)

            fig, axs = plt.subplots(1, 2)

            plt_images = []
            plt_images.append(axs[0].imshow(np.rot90(image.numpy(), k=-1)))
            axs[0].title.set_text("Original")

            # 1:  Flip upside down
            # 2:  Flip left right
            tri_coin_flip = random.sample((0, 1, 2), 1)[0]
            if tri_coin_flip == 1:
                image = ants.from_numpy(np.fliplr(image.numpy()), origin=image.origin, spacing=image.spacing, direction=image.direction)
            elif tri_coin_flip == 2:
                image = ants.from_numpy(np.flipud(image.numpy()), origin=image.origin, spacing=image.spacing, direction=image.direction)
            elif tri_coin_flip == 3:
                image = ants.from_numpy(np.fliplr(np.flipud(image.numpy())), origin=image.origin, spacing=image.spacing, direction=image.direction)

            image = (image - image.min()) / (image.max() - image.min())
            batchX[0,:,:,0] = image.numpy()
            batchY = model.predict(batchX, verbose=False)
            print(str(i), ": flip = ", str(tri_coin_flip), ", predicted = ", batchY)            
            plt_images.append(axs[1].imshow(np.rot90(image.numpy(), k=-1)))            
            predicted = np.argmax(batchY)
            axs[1].title.set_text("Flipped " + str(tri_coin_flip) + "-->" + str(predicted))
            fig.suptitle(batchY)
            
            if int(predicted) != int(tri_coin_flip):
                print(image_file)
                plt.show()
            else:    
                plt.close()



