import ants
import antspynet
import imageio
import numpy as np

brain = ants.image_read("r16slice.nii.gz")
starry_night = imageio.imread("starry_night.jpg")
starry_night_red = np.squeeze(starry_night[:,:,0])
starry_night_blue = np.squeeze(starry_night[:,:,1])
starry_night_green = np.squeeze(starry_night[:,:,2])

starry_night_ants = ants.merge_channels([
                        ants.from_numpy(starry_night_red.astype(float)),
                        ants.from_numpy(starry_night_blue.astype(float)),
                        ants.from_numpy(starry_night_green.astype(float))])
starry_night_ants.components = 3                           

neural = antspynet.neural_style_transfer( 
           brain,
           starry_night_ants,
           initial_combination_image=None,
           number_of_iterations=100,
           learning_rate=10.0,
           total_variation_weight=8.5e-5,
           content_weight=0.025,
           style_image_weights=1.0,
           content_layer_names=[
           'block5_conv2'],
           style_layer_names="all",
           content_mask=None,
           style_masks=None,
           use_shifted_activations=True,
           use_chained_inference=False,
           verbose=True,
           output_prefix="/Users/ntustison/Desktop/NeuralTransferStyle/testPy")