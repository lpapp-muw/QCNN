#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:32:15 2025

@author: smoradi
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.metrics import confusion_matrix
import numpy as np
from glob import glob
from os.path import basename, join
import SimpleITK as sitk
from natsort import natsorted
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
import numpy as np
import random
import os


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

np.set_printoptions(threshold=np.inf)

#92.80124, 97.65078

# Load files function (same as in your provided code)
def load_files(pet_paths, seg_paths, max_value=92.80124):
    images = []
    labels = []
    for pet_file, seg_file in zip(pet_paths, seg_paths):
        sitk_img = sitk.ReadImage(pet_file)
        sitk_seg = sitk.ReadImage(seg_file)

        img_arr = sitk.GetArrayFromImage(sitk_img)
        seg_arr = sitk.GetArrayFromImage(sitk_seg)

        masked_arr = np.where(seg_arr == 1, img_arr, 0)
        masked_arr = np.clip(masked_arr / max_value, 0.0, 1.0)
        #masked_arr = masked_arr.flatten()

        label = basename(pet_file).split(".")[0]
        label = int(label[-1])

        images.append(masked_arr)
        labels.append(label)

    return np.asarray(images), np.asarray(labels)

# Nifti loader function (same as in your provided code)
def nifti_loader(path_to_execution):
    train_dir = join(path_to_execution, "Train")
    test_dir = join(path_to_execution, "Test")

    train_files_pet = natsorted(glob(join(train_dir, "*", "*PET*nii.gz")))
    train_files_seg = natsorted(glob(join(train_dir, "*", "*mask*nii.gz")))

    test_files_pet = natsorted(glob(join(test_dir, "*", "*PET*nii.gz")))
    test_files_seg = natsorted(glob(join(test_dir, "*", "*mask*nii.gz")))

    train_images, train_labels = load_files(train_files_pet, train_files_seg)
    test_images, test_labels = load_files(test_files_pet, test_files_seg)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }

def create_cnn_model(learning_rate, dropout_rate):
    model = models.Sequential([
        layers.Input(shape=(32, 32, 32, 1)),  # Explicit Input layer
        layers.Conv3D(32, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(128, (3, 3, 3), activation='relu'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),  # Dropout as an optimized parameter for fitting
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# Training and evaluation function

def train_cnn_and_evaluate(n_reps, initial_lr, dropout_rate):
    test_accuracies = []
    #test_pro = []
    BB = []
    conf_mat = []

    for _ in range(n_reps):
        path_to_execution = "/home/smoradi/anaconda3/envs/myenv/Test/3layer/Executions_done/EXECUTIONS/E3"
        #path_to_execution = "/home/smoradi/anaconda3/envs/myenv/Test/3layer/Executions_done/HARMONIZED_EXECUTIONS/EXECUTIONS/E1"
        data = nifti_loader(path_to_execution)
        x_train = data["train_images"]
        y_train = data["train_labels"]
        x_test = data["test_images"]
        y_test = data["test_labels"]

        learning_rate = initial_lr

        model = create_cnn_model(learning_rate, dropout_rate)

        model.fit(x_train, y_train, epochs=n_epochs, verbose=0, validation_data=(x_test, y_test))

        train_acc = model.evaluate(x_train, y_train, verbose=0)[1]
        test_acc = model.evaluate(x_test, y_test, verbose=0)[1]

        test_accuracies.append(test_acc)

        train_probs = model.predict(x_train)
        test_probs = model.predict(x_test)
        #test_pro.append(test_probs)

        train_predictions = np.round(train_probs).astype(int)
        test_predictions = np.round(test_probs).astype(int)

        #test_confusion_matrix = confusion_matrix(y_test, test_predictions)

        #BB.append(test_predictions)

        #conf_mat.append(test_confusion_matrix)

    return test_predictions

# Hyperparameters
n_epochs =100
n_reps = 1
initial_lr = 0.005  
dropout_rate = 0.1

# Training and evaluation
test_predictions = train_cnn_and_evaluate(n_reps, initial_lr, dropout_rate)

test_dir = "/home/smoradi/anaconda3/envs/myenv/Test/3layer/Executions/EXECUTIONS/E3/Test"
test_files_pet = natsorted(glob(join(test_dir, "*", "*PET*nii.gz")))
test_patient_codes = [os.path.basename(f).split("_")[0] for f in test_files_pet]


df_results = pd.DataFrame({
    "Patient Code": test_patient_codes,
    "test_predictions": test_predictions,
})

output_file = "predicted_labels.xlsx"
df_results.to_excel(output_file, index=False)
print(f"Saved test results to: {output_file}")
