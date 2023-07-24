import ants
import antspynet
import deepsimlr
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import glob

import torch
import torchinfo
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import numpy as np
import random

torch.device(3)

base_directory = '/home/ntustison/Data/MRIModalityClassification/'
scripts_directory = base_directory + 'Scripts/'
data_directory = base_directory
brats_directory = '/home/ntustison/Data/BRATS/TCIA/'

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

modalities = ("T1", "T2", "FLAIR", "T2Star", "Mean DWI", "Mean Bold", "ASL perfusion")

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

weights_filename = scripts_directory + "mri_modality_classification_pytorch.h5"
if os.path.exists(weights_filename):
    model.load_state_dict(torch.load(weights_filename))

# model2 = antspynet.create_resnet_model_3d((None, None, None, channel_size),
#                                           number_of_classification_labels=number_of_classification_labels,
#                                           mode="classification",
#                                           layers=(1, 2, 3, 4),
#                                           residual_block_schedule=(3, 4, 6, 3),
#                                           lowest_resolution=64,
#                                           cardinality=1,
#                                           squeeze_and_excite=False)



# model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4),
#               loss=tf.keras.losses.CategoricalCrossentropy())

################################################
#
#  Load the data
#
################################################

print("Loading brain data.")

t1_images = (*glob.glob(data_directory + "**/*T1w.nii.gz", recursive=True),
             *glob.glob(brats_directory + "**/*T1*.nii.gz", recursive=True))
t2_images = (*glob.glob(data_directory + "**/*T2w.nii.gz", recursive=True),
             *glob.glob(brats_directory + "**/*T2*.nii.gz", recursive=True))
flair_images = (*glob.glob(data_directory + "**/*FLAIR.nii.gz", recursive=True),
                *glob.glob(brats_directory + "**/*FLAIR*.nii.gz", recursive=True))
t2star_images = glob.glob(data_directory + "**/*T2starw.nii.gz", recursive=True)
dwi_images = glob.glob(data_directory + "**/*MeanDwi.nii.gz", recursive=True)
bold_images = glob.glob(data_directory + "**/*MeanBold.nii.gz", recursive=True)
perf_images = glob.glob(data_directory + "**/*asl.nii.gz", recursive=True)

images = list()
images.append(t1_images)
images.append(t2_images)
images.append(flair_images)
images.append(t2star_images)
images.append(dwi_images)
images.append(bold_images)
images.append(perf_images)

for i in range(len(images)):
    print("Number of", modalities[i], "images: ", len(images[i]))


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
                 number_of_samples=1):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_files = image_files
        self.template = template
        self.number_of_samples = number_of_samples

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        number_of_classes = 7  
        random_modality = random.sample(list(range(number_of_classes)), 1)[0]
        random_index = random.sample(list(range(len(self.image_files[random_modality]))), 1)[0]

        image = ants.image_read(self.image_files[random_modality][random_index])

        center_of_mass_template = ants.get_center_of_mass(template*0 + 1)
        center_of_mass_image = ants.get_center_of_mass(image*0 + 1)
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_template), translation=translation)
        image = ants.apply_ants_transform_to_image(xfrm, image, template)

        if random.uniform(0.0, 1.0) < 0.9:
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

        image_array = np.expand_dims(image.numpy(), axis=-1)

        # swap color axis because
        # numpy image: H x W x D x C
        # torch image: C x H x W x D

        image_array = image_array.transpose((3, 0, 1, 2))
        image_tensor = torch.from_numpy(image_array)

        modality = int(random_modality)

        return image_tensor, modality

transformed_dataset = MRIDataset(image_files=images,
                                 template=template,
                                 number_of_samples=32*16)
transformed_dataset_testing = MRIDataset(image_files=images,
                                        template=template,
                                        number_of_samples=16)
train_dataloader = DataLoader(transformed_dataset, batch_size=16,
                        shuffle=True, num_workers=4)
test_dataloader = DataLoader(transformed_dataset_testing, batch_size=16,
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
        start = time.time()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.squeeze())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
         
        loss, current = loss.item(), (batch + 1) * len(X)
        end = time.time()
        elapsed = end - start
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], elapsed: {elapsed:>5f} seconds.")


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
    return correct


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

current_accuracy = 0.0

epochs = 200
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    accuracy = test_loop(test_dataloader, model, loss_fn)
    if accuracy > current_accuracy:
        print("Accuracy improved.")
        torch.save(model.state_dict(), weights_filename)
        current_accuracy = accuracy
print("Done!")

