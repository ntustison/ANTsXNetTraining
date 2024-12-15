import ants
import antspynet
import glob
import random
import os
import numpy as np

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = "8"

base_directory = "/Users/ntustison/Data/Public/HCP/DeepLearning/HoaTraining/"
data_directory = base_directory + "SubcorticalParcellations/dseg/"

t1_files = glob.glob(data_directory + "*.nii.gz")

number_of_simulations = 100
write_images_to_disk = False

labels = tuple(range(1, 33))
                        
probability_images = [None] * len(labels)


for i in range(number_of_simulations):
    print("Simulation " + str(i))
    random_index = random.randint(0, len(t1_files)-1)
    
    t1_file = t1_files[random_index]
    labels_file = t1_files[random_index]
    
    t1 = ants.image_read(t1_file)
    label_image = ants.image_read(labels_file)
    
    random_data = antspynet.randomly_transform_image_data(
        t1, [[t1]],
        segmentation_image_list=[label_image],
        number_of_simulations=1,
        transform_type="deformation",
        sd_affine=0.01,
        deformation_transform_type="bspline",
        number_of_random_points=1000,
        sd_noise=10.0,
        number_of_fitting_levels=4,
        mesh_size=1,
        sd_smoothing=4.0,
        input_image_interpolator="linear",
        segmentation_image_interpolator="nearestNeighbor"
        )
    
    warped = ants.image_clone(random_data['simulated_segmentation_images'][0])
    if write_images_to_disk:
        ants.image_write(warped, base_directory + "warped_" + str(i) + ".nii.gz")

    if i == 0:
        for j in range(len(labels)):
            print("   Label " + str(labels[j]))
            label_image = ants.smooth_image(ants.threshold_image(warped, labels[j], labels[j], 1, 0), 1)
            probability_images[j] = label_image
    else:
        for j in range(len(labels)):
            print("   Label " + str(labels[j]))
            label_image = ants.smooth_image(ants.threshold_image(warped, labels[j], labels[j], 1, 0), 1)
            probability_images[j] = (probability_images[j] * i + label_image) / (i + 1)

    for j in range(len(labels)):
        ants.image_write(probability_images[j], base_directory + "PriorProbabilityImages/prior" + str(labels[j]) + ".nii.gz")


