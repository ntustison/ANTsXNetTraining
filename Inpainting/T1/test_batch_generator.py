import ants
import antspynet

import numpy as np
import glob

from batch_inpainting_generator import batch_generator


t1_images = glob.glob("/Users/ntustison/Data/Public/Kirby/T1/*.nii.gz")
image_size = (256, 266)
channel_size = 1

template = ants.image_read(antspynet.get_antsxnet_data("oasis"))
template_labels = ants.image_read("dktWithWhiteMatterLobes.nii.gz")
template_roi = ants.image_read("brainMaskDilated.nii.gz")

batch_size = 16

generator = batch_generator(batch_size=batch_size,
                            t1s=t1_images,
                            image_size=(image_size[0], image_size[1]),
                            number_of_channels=channel_size,
                            template=template,
                            template_labels=template_labels,
                            template_roi=template_roi,
                            add_2d_masking=True,
                            do_histogram_intensity_warping=False,
                            do_simulate_bias_field=False,
                            do_add_noise=False,
                            do_data_augmentation=False,
                            return_ones_masks=False
                            )

x = next(generator)
ants.image_write(ants.from_numpy(x[0][0]), "X.nii.gz")
ants.image_write(ants.from_numpy(x[0][1]), "XMask.nii.gz")
ants.image_write(ants.from_numpy(x[1]), "Y.nii.gz")
