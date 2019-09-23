"""
Raw implementation of training and validation script using pytorch for the task of flixstock Jam
"""

import os
import gc
import subprocess
import time

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

from modules import helpers
from modules import data

__author__ = "Vinay Kumar"
__copyright__ = "@vinaykumar2491, 2019"
__project__ = "flixstockJam"

## Initial setup
gc.collect()                                        ## garbage collection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

data_preprocessing = True
do_training = True
do_testing = False
root_path = pathlib.Path(f"../../data")
data_path = pathlib.Path(f"{root_path}/images")
attr_csv_path = pathlib.Path(f"{root_path}/attributes.csv")

df = pd.DataFrame()
if data_preprocessing:
    df = helpers.preprocess_from_csv(data_path=data_path, attr_path=attr_csv_path, data_augmentation=False,
                        multilabel_classification=True, target_attr="attr", multilabel_start=1,
                        multilabel_end=4, multilabel_delim=" : ")
## convert the input and labels into tensors


if do_training:
    ## Step-1: Preparing the training and validation dataset
    ## Step-2: Preparing the flixNet architecture
    ## Step-3: Designing the training and validation hyperparams, metrics etc
    ## Step-4: Training and validating the flixNet
    ## Step-5: Analyzing the train/valid metrics
    ## Step-6: Saving the model, so as to be used during testing.
    pass

if do_testing:
    ## Step-1: Prepare the test dataset
    ## Step-2: Load the pretrained model
    ## Step-3: Define the evaluation metrics
    ## Step-5: Test the pretrained model against test dataset
    ## Step-6: Record the results
    pass




