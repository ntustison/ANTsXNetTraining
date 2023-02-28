import ants
import antspynet

import numpy as np
import glob

from batch_inpainting_patch_generator import batch_generator


t1_images = glob.glob("/Users/ntustison/Data/Public/Kirby/T1/*.nii.gz")
image_size = (256, 256, 256)
channel_size = 1

template = ants.image_read(antspynet.get_antsxnet_data("oasis"))
template_labels = ants.image_read("dktWithWhiteMatterLobes.nii.gz")
template_roi = ants.image_read("brainMaskDilated.nii.gz")

template_image_size = (256, 256, 256)
template = antspynet.pad_or_crop_image_to_size(template, template_image_size)
template_labels = antspynet.pad_or_crop_image_to_size(template_labels, template_image_size)
template_roi = antspynet.pad_or_crop_image_to_size(template_roi, template_image_size)

batch_size = 4

generator = batch_generator(batch_size=batch_size,
                            t1s=t1_images,
                            patch_size=(64, 64, 64),
                            number_of_channels=channel_size,
                            template=template,
                            template_labels=template_labels,
                            template_roi=template_roi,
                            add_2d_masking=True,
                            do_histogram_intensity_warping=False,
                            do_simulate_bias_field=False,
                            do_add_noise=False,
                            do_data_augmentation=False,
                            return_ones_masks=False,
                            verbose=True
                            )

x = next(generator)

for i in range(batch_size):
    ants.image_write(ants.from_numpy(np.squeeze(x[0][0][i,:,:,:,0])), "PatchImages/X" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(x[0][1][i,:,:,:,0])), "PatchImages/XMask" + str(i) + ".nii.gz")
    ants.image_write(ants.from_numpy(np.squeeze(x[1][i,:,:,:,0])), "PatchImages/Y" + str(i) + ".nii.gz")

# ants.image_write(ants.from_numpy(x[0][0]), "X.nii.gz")
# ants.image_write(ants.from_numpy(x[0][1]), "XMask.nii.gz")
# ants.image_write(ants.from_numpy(x[1]), "Y.nii.gz")
