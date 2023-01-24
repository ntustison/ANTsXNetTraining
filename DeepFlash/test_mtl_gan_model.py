import ants
import antspynet
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import glob

input_image_size = (64, 64, 96, 1)

from create_pix2pix_gan_model import Pix2PixGanModel

base_directory = "/home/ntustison/Data/DeepFlash/Data/"

adni_files = glob.glob(base_directory + "NiftiRois*/ADNI/*.nii.gz")
adrc_files = glob.glob(base_directory + "NiftiRois*/ADRC/*.nii.gz")
ixi_files = glob.glob(base_directory + "NiftiRois*/IXI/*.nii.gz")
kirby_files = glob.glob(base_directory + "NiftiRois*/Kirby/*.nii.gz")
nki_files = glob.glob(base_directory + "NiftiRois*/NKI/*.nii.gz")
oasis_files = glob.glob(base_directory + "NiftiRois*/OASIS/*.nii.gz")
oasis3_files = glob.glob(base_directory + "NiftiRois*/OASIS3/*.nii.gz")
srpb_files = glob.glob(base_directory + "NiftiRois*/SRPB1600/*.nii.gz")
stark_files = glob.glob(base_directory + "NiftiRois*/StarkTrainingSet/*.nii.gz")

source_files = (*adni_files, *adrc_files, *ixi_files, *kirby_files,
                *nki_files, *oasis_files, *oasis3_files, *srpb_files)
target_files = stark_files

############################

number_of_source_files = len(source_files)
X_source = np.zeros(shape=(number_of_source_files, *input_image_size))

print("Reading source files (n=" + str(number_of_source_files) + ")")

for i in range(number_of_source_files):
   image = ants.image_read(source_files[i])
   X_source[i,:,:,:,0] = image.numpy()

print("Done.")


number_of_target_files = len(target_files)
X_target = np.zeros(shape=(number_of_target_files, *input_image_size))

print("Reading target files (n=" + str(number_of_target_files) + ")")

for i in range(number_of_target_files):
   image = ants.image_read(target_files[i])
   X_target[i,:,:,:,0] = image.numpy()

print("Done.")


gan_model = Pix2PixGanModel(input_image_size=input_image_size)

gan_model.train(X_source, X_target, number_of_epochs=3000, sample_interval=250,
  batch_size=16, output_prefix="./GanOutput/deepFlash" )


