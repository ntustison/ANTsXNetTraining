import ants
import antspynet
import random

image = ants.image_read("/Users/ntustison/Data/Public/XRayCT/Data/images_01/00000001_000.png")
ants.image_write(image, "~/Desktop/image.nii.gz")                                                   

for i in range(10):
    sd_histogram_warping = 0.01
    break_points = [0.2, 0.4, 0.6, 0.8]
    displacements = list()
    for b in range(len(break_points)):
        displacements.append(random.gauss(0, sd_histogram_warping))
    image = antspynet.histogram_warp_image_intensities(image,
                                                       break_points=break_points,
                                                       clamp_end_points=(True, True),
                                                       displacements=displacements)
    data_aug = antspynet.randomly_transform_image_data(image,
                                                       [[image]], 
                                                       segmentation_image_list=None,
                                                       number_of_simulations=1,
                                                       transform_type="affine",
                                                       sd_affine=0.025,
                                                       deformation_transform_type="bspline",
                                                       number_of_random_points=50,
                                                       sd_noise=2.0,
                                                       mesh_size=4
                                                       )
    image = data_aug['simulated_images'][0][0]
    ants.image_write(image, "~/Desktop/image_aug" + str(i) + ".nii.gz")

