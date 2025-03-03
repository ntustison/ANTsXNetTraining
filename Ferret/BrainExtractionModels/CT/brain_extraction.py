import ants
import antspynet
import tensorflow as tf
import numpy as np
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

template_url = "https://figshare.com/ndownloader/files/52744973"
weights_url = "https://figshare.com/ndownloader/files/52745060"

template_file = tf.keras.utils.get_file("ferretCTTemplate.nii.gz", template_url)
template = ants.image_read(template_file)
template = antspynet.pad_or_crop_image_to_size(template, (256, 256, 224))

# Just clone the template to simplify the example
image = ants.image_clone(template)
image = ants.add_noise_to_image(image, noise_model="additivegaussian", noise_parameters = (0.0, 0.01))

weights_file = tf.keras.utils.get_file("ferretCtBrainExtraction3D.weights.h5", weights_url)

print("Warping to CT ferret template.")

center_of_mass_reference = ants.get_center_of_mass(template * 0 + 1)
center_of_mass_image = ants.get_center_of_mass(image * 0 + 1)
translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
    center=np.asarray(center_of_mass_reference), translation=translation)
xfrm_inv = ants.invert_ants_transform(xfrm)

image_warped = ants.apply_ants_transform_to_image(xfrm, image, template, interpolation="linear")
image_warped = ants.iMath_normalize(image_warped)

print("Create model and load weights.")

unet_model = antspynet.create_unet_model_3d((*template.shape, 1),
                                   number_of_outputs=2, mode="classification",
                                   number_of_filters=(16, 32, 64, 128),
                                   convolution_kernel_size=(3, 3, 3),
                                   deconvolution_kernel_size=(2, 2, 2))
unet_model.load_weights(weights_file)

batchX = np.zeros((1, *template.shape, 1))
batchX[0,:,:,:,0] = image_warped.numpy()

print("Prediction.")
batchY = np.squeeze(unet_model.predict(batchX, verbose=True)[0,:,:,:,1])

probability_mask = ants.from_numpy(batchY, origin=image_warped.origin,
                                   spacing=image_warped.spacing, direction=image_warped.direction)
probability_mask =  ants.apply_ants_transform_to_image(xfrm_inv, probability_mask,
                                                       image, interpolation="linear")

ants.image_write(probability_mask, "./brain_extraction_probability_mask.nii.gz")
