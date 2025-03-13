import ants
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "True"
import tensorflow as tf

from create_normalizing_flow_model import create_normalizing_flow_model

image_shape = (256, 256)

weights_filename = "nvp_t1_axial.weights.h5"
nvp_model = create_normalizing_flow_model((*image_shape, 1), 
    hidden_layers=[512, 512], flow_steps=6, regularization=0.0,
    validate_args=False)
nvp_model.load_weights(weights_filename)

image0 = ants.image_read("1150497_T1w_slice150.nii.gz")
image0 = ants.iMath_normalize(image0)

image1 = ants.image_read("sub-0409_T1w_slice137.nii.gz")
image1 = ants.iMath_normalize(image1)

batchX = np.zeros((2, *image_shape, 1))
batchX[0,:,:,0] = image0.numpy()
batchX[1,:,:,0] = image1.numpy()

batchY = nvp_model.call(batchX.astype('float32'))
Y0 = batchY.numpy()[0,:]
Y1 = batchY.numpy()[1,:]
distance = np.linalg.norm(Y1 - Y0)
Ymid = Y0 + (Y1 - Y0) * 0.5 

X0 = nvp_model.inverse(tf.expand_dims(batchY[0,:], axis=0))
X1 = nvp_model.inverse(tf.expand_dims(batchY[1,:], axis=0))
Xmid = nvp_model.inverse(tf.expand_dims(Ymid, axis=0))

ants.image_write(ants.from_numpy_like(np.squeeze(X0.numpy()), image0), "X0.nii.gz")
ants.image_write(ants.from_numpy_like(np.squeeze(X1.numpy()), image0), "X1.nii.gz")
ants.image_write(ants.from_numpy_like(np.squeeze(Xmid.numpy()), image0), "Xmid.nii.gz")
