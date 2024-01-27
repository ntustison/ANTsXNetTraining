import ants
import antspynet
import deepsimlr
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
import glob

base_directory = '/home/ntustison/Data/reproduce-chexnet/'
image_directory = base_directory + "data/nifti/"

disease_categories = ['Atelectasis',
                        'Cardiomegaly',
                        'Effusion',
                        'Infiltration',
                        'Mass',
                        'Nodule',
                        'Pneumonia',
                        'Pneumothorax',
                        'Consolidation',
                        'Edema',
                        'Emphysema',
                        'Fibrosis',
                        'Pleural_Thickening',
                        'Hernia']
number_of_dx = len(disease_categories)

demo = pd.read_csv(base_directory + "nih_labels.csv", index_col=0)
test_demo = demo.loc[demo['fold'] == 'test']

pytorch_chexnet_predictions = np.zeros((test_demo.shape[0], number_of_dx))
keras_chexnet_predictions = np.zeros((test_demo.shape[0], number_of_dx))
antsxnet_chexnet_predictions = np.zeros((test_demo.shape[0], number_of_dx))
true_diagnoses = np.zeros((test_demo.shape[0], number_of_dx))

for i in range(test_demo.shape[0]):
    print(i, " out of ", test_demo.shape[0])    
    subject_row = demo.iloc[[i]]
    base_image_file = subject_row.index.values[0]
    base_image_file = base_image_file.replace(".png", ".nii.gz")
    image_file = glob.glob("/home/ntustison/Data/reproduce-chexnet/data/nifti/" + base_image_file)[0]    
    mask_file = image_file.replace("nifti", "masks")
    if not os.path.exists(image_file) or not os.path.exists(mask_file):
        print(image_file, " does not exist.")
        continue
    image = ants.image_read(image_file)
    mask = ants.image_read(mask_file)

    cxr_pytorch = deepsimlr.chexnet(image, verbose=False)      
    cxr_keras = antspynet.chexnet(image, use_antsxnet_variant=False, verbose=False)      
    cxr_antsxnet = antspynet.chexnet(image, lung_mask=mask, use_antsxnet_variant=True, verbose=False)      

    for d in range(number_of_dx):
        pytorch_chexnet_predictions[i,d] = cxr_pytorch[disease_categories[d]]
        keras_chexnet_predictions[i,d] = cxr_keras[disease_categories[d]]
        antsxnet_chexnet_predictions[i,d] = cxr_antsxnet[disease_categories[d]]
        true_diagnoses[i,d] = subject_row[disease_categories[d]]

    if i > 0 and i % 1000 == 0:        
        pytorch_auc = np.zeros((number_of_dx,))
        keras_auc = np.zeros((number_of_dx,))
        antsxnet_auc = np.zeros((number_of_dx,))
        for j in range(number_of_dx):
            pytorch_auc[j] = sklm.roc_auc_score(true_diagnoses[:i,j], pytorch_chexnet_predictions[:i,j])            
            keras_auc[j] = sklm.roc_auc_score(true_diagnoses[:i,j], keras_chexnet_predictions[:i,j])            
            antsxnet_auc[j] = sklm.roc_auc_score(true_diagnoses[:i,j], antsxnet_chexnet_predictions[:i,j])            
        print("Pytorch:  ", pytorch_auc)
        print("Keras:  ", keras_auc)
        print("ANTsXNet:  ", antsxnet_auc)

np.save("chexnet_pytorch_predictions.npy", pytorch_chexnet_predictions)
np.save("chexnet_keras_predictions.npy", keras_chexnet_predictions)
np.save("chexnet_antsxnet_predictions.npy", antsxnet_chexnet_predictions)
np.save("chexnet_true_diagnoses.npy", true_diagnoses)

    
