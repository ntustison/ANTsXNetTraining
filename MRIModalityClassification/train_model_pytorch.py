import ants
import antspynet
import deepsimlr

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import glob

import torch
import torchinfo
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np
import random

base_directory = '/home/ntustison/Data/MRIModalityClassification/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory + "Nifti/"

image_size = (112, 112, 112)
resample_size = (2, 2, 2)

template = ants.image_read(antspynet.get_antsxnet_data("kirby"))
template = ants.resample_image(template, resample_size)
template = antspynet.pad_or_crop_image_to_size(template, image_size)
direction = template.direction
direction[0, 0] = 1.0
ants.set_direction(template, direction)
ants.set_origin(template, (0, 0, 0))

################################################
#
#  Create the model and load weights
#
################################################

# modalities:
#     T1
#     T2
#     FLAIR
#     T2Star
#     Mean DWI
#     Mean Bold
#     ASL perfusion

number_of_classification_labels = 7
channel_size = 1

model = deepsimlr.create_resnet_model_3d(input_channel_size=channel_size,
                                         number_of_classification_labels=number_of_classification_labels,
                                         mode="classification",
                                         layers=(1, 2, 3, 4),
                                         residual_block_schedule=(3, 4, 6, 3),
                                         lowest_resolution=64,
                                         cardinality=1,
                                         squeeze_and_excite=False)
# torchinfo.summary(model, input_size=(1, 1, *image_size))

# model2 = antspynet.create_resnet_model_3d((None, None, None, channel_size),
#                                           number_of_classification_labels=number_of_classification_labels,
#                                           mode="classification",
#                                           layers=(1, 2, 3, 4),
#                                           residual_block_schedule=(3, 4, 6, 3),
#                                           lowest_resolution=64,
#                                           cardinality=1,
#                                           squeeze_and_excite=False)

# weights_filename = scripts_directory + "mri_modality_classification_pytorch.h5"

# if os.path.exists(weights_filename):
#     model.load_weights(weights_filename)

# model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
#               loss=tf.keras.losses.CategoricalCrossentropy())

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = glob.glob(data_directory + "**/*T1w.nii.gz", recursive=True)
t2_images = glob.glob(data_directory + "**/*T2w.nii.gz", recursive=True)
flair_images = glob.glob(data_directory + "**/*FLAIR.nii.gz", recursive=True)
t2star_images = glob.glob(data_directory + "**/*T2starw.nii.gz", recursive=True)
dwi_images = glob.glob(data_directory + "**/*MeanDwi.nii.gz", recursive=True)
bold_images = glob.glob(data_directory + "**/*MeanBold.nii.gz", recursive=True)
perf_images = glob.glob(data_directory + "**/*asl.nii.gz", recursive=True)

images = t1_images + t2_images + flair_images + t2star_images + dwi_images + bold_images + perf_images
modalities = np.concatenate((
             np.zeros((len(t1_images),), dtype=np.int8),
             np.zeros((len(t2_images),), dtype=np.int8) + 1,
             np.zeros((len(flair_images),), dtype=np.int8) + 2,
             np.zeros((len(t2star_images),), dtype=np.int8) + 3,
             np.zeros((len(dwi_images),), dtype=np.int8) + 4,
             np.zeros((len(bold_images),), dtype=np.int8) + 5,
             np.zeros((len(perf_images),), dtype=np.int8) + 6),
             dtype=np.int8
             )

print( "Training")

################################################
#
# Set up the data loader
#
################################################

class MRIDataset(Dataset):
    """MRI dataset."""

    def __init__(self,
                 image_files,
                 template,
                 modalities,
                 transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_files = image_files
        self.template = template
        self.modalities = modalities
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        random_index = random.sample(list(range(len(self.image_files))), 1)[0]

        image = ants.image_read(self.image_files[random_index])

        center_of_mass_template = ants.get_center_of_mass(template*0 + 1)
        center_of_mass_image = ants.get_center_of_mass(image*0 + 1)
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_template), translation=translation)
        image = ants.apply_ants_transform_to_image(xfrm, image, template)

        if random.uniform(0.0, 1.0) < 0.75:
            noise_model = None
            if random.uniform(0.0, 1.0) < 0.33:
                noise_model = ("additivegaussian", "shot", "saltandpepper")
            data_aug = antspynet.data_augmentation(input_image_list=[[image]],
                                                segmentation_image_list=None,
                                                pointset_list=None,
                                                number_of_simulations=1,
                                                reference_image=template,
                                                transform_type='affineAndDeformation',
                                                noise_model=noise_model,
                                                sd_simulated_bias_field=1.0,
                                                sd_histogram_warping=0.05,
                                                sd_affine=0.05,
                                                output_numpy_file_prefix=None,
                                                verbose=False)
            image = data_aug['simulated_images'][0][0]
        image = (image - image.min()) / (image.max() - image.min())

        modality = np.zeros((7,))
        modality[self.modalities[random_index]] = 1

        sample = {'image': image.numpy(), 'modality': modality}

        if self.transform:
            sample = self.transform(sample)

        return sample

transformed_dataset = MRIDataset(image_files=images,
                                 template=template,
                                 modalities=modalities,
                                 transform=None)
train_dataloader = DataLoader(transformed_dataset, batch_size=16,
                        shuffle=True, num_workers=4)
test_dataloader = DataLoader(transformed_dataset, batch_size=16,
                        shuffle=True, num_workers=4)

###
#
# Optimize
#

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

epochs = 200
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")