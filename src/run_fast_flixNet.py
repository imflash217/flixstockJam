"""Using fastai to solve the multi-label classification problem in fashion detection flixstock DB.
I am using a pretrained resnet architecture combined with BCEWithLogitsLoss loss criterion.
The training schedule is carried using one-cycle-training process published by Leslie Smith in 2017.
A jupyter-notebook version of this solution is also available in the repo.
This codebase is compatible with Python 3.6+ only
"""

from fastai import *
from fastai.vision import *
import pandas as pd
import gc

__author__ = "Vinay Kumar"
__copyright__ = "@vinaykumar2491, 2019"
__project__ = "flixstockJam"

## Step-1: Load and analyze the database

data_path = Path(f"../../data")     ## This has type PosixPath which is conveniently interpreted by pandas, fastai, pytorch.
df = pd.read_csv(filepath_or_buffer=data_path/"attributes.csv")
print(f"{df.head()}")               ## viewing and analyzing the provided data/labels & its structure

## Step-2: PREPROCESSING the database
## After looking at the dataframe, I am restructuring the data representaion to convenience as:
## 2.a. Check for any NaN in the labels (neck, ...).
## 2.b. Create a separate label for each item representing a boolean if the item has NaN for the
##      respective labels. Create separate new labels for each existing label types (eg. neck_NaN).
## 2.c. Replace the NaN values in the original label types with the mean of that label type in DB.
## 2.d. Concat all the original labels for an item and save as a new label called "attr". We will be
##      using this "attr" as a label for the respective image item to train our neuralNet.
## 2.e. Creating the dataset which can be used by the learner

orig_new_ref = {}
label_delim = " : "
batch_size = 16
epochs = 4

for lbl in df.columns[1:]:
    ## Step 2.a, 2.b
    df[f"{lbl}_NaN"] = (df[lbl].isnull().astype(float))
    ## Step 2.c
    mean = df[lbl].mean()
    df[lbl].fillna(value=round(mean), inplace=True)
    unique_vals = sorted(df[lbl].dropna().unique())
    orig_new_ref[lbl] = {}
    orig_new_ref[lbl]["orig"] = unique_vals
    orig_new_ref[lbl]["new"] = [f"{lbl}{int(x)}" for x in unique_vals]
    df[lbl].replace(orig_new_ref[lbl]["orig"], orig_new_ref[lbl]["new"], inplace=True)

## Step 2.d
df["attr"] = ""
for i in df.columns[1:4]:
    ## Step 2.d
    if df["attr"].all() == "":
        df["attr"] = df[i]
    else:
        df["attr"] = df["attr"] + label_delim + df[i]

## Sanity check: remove the files from csv that donot exist in images
df.drop(df[df.filename=="11495010560702-The-Indian-Garage-Co-Men-Tshirts-7851495010560569-2.jpg"].index, inplace=True)

## Step 2.e
transforms = get_transforms()
src = (ImageList.from_df(df=df, path=data_path, cols=["filename"], folder="images")
       .split_by_rand_pct(0.2)                  ## splits train:valid into 80:20 %age ratio
       .label_from_df(cols="attr", label_delim=label_delim))
data = (src.transform(transforms)               ## transforms like flip, rotate, blur, resize etc. I'm just using the default horizontal flip
        .databunch(bs=batch_size)               ## batchsize to load the data from dataloader
        .normalize(imagenet_stats))             ## normalizing using imagenet stats as I'll be using a pretrianed model parameters
## data.show_batch(rows=3)
## print(data.c)                                ## total number of classes in labels
## print(data.classes)                          ## list of all classes in labels
gc.collect                                      ## garbage collection to release memory

## Step-3: Load the neuralNet architecture & defining the metrics before creating a learner
flixNet_arch = models.resnet50                  ## downloading the arch and its pretrained parameters

## defining the evaluation metrics (but its not used for training rather only for us to judge the performance of flixNet)
## As, no metric is suggested in the problem statement, I'm using thresholded accuracy
metric = partial(accuracy_thresh, thresh=0.1)   ## Decinding upon what threshold to use will require more experiments on this dataset & flixNet

## Step-4: Train the neuralNet arch
learner = cnn_learner(data, flixNet_arch, metrics=metric)
print(learner)                                  ## displays the whole learner arch & loss function
                                                ## inherently I am using BCEWithLogitsLoss as loss criterion
                                                ## b'coz its serves the purpose of multilabel classification unlike BCE (BinaryCssEntropy) loss
learner.lr_find()                               ## superb method to decide upon the range of learning rates to choose from
                                                ## lr_find() perfmorms better than GridSearch or RandoSearch usually used in finding lr.
learner.recorder.plot()                         ## plots the lrs against the loss generated. Use it to decide the range of lr.
learner.loss_func                               ## shows the loss func used. (here, BCEWithLogitsLoss)

## I'm using cyclical training starategy developed by Leslie Smith, with distributed learning rates across the layers of flixNet
## Currently the flixNet layers are frozen except for last layer added to pretrained resnet to suit out multi-hot-encoded labels
learner.fit_one_cycle(cyc_len=epochs, max_lr=slice(1e-3, 1e-1))     ## training the last layers of flixNet ONLY

learner.recorder.plot_losses()                  ## plot the train & valid losses
learner.recorder.plot_metrics()                 ## plot the metrics (here, accuracy_thresh)
learner.show_results()                          ## results from the validation dataset showing the true & predicted labels
learner.save("stage-1")

## Step-5: Analyze the train/valid losses and retrain the neuralNet
## Unfreezing all the layers of pretrained resnet, to train it on our dataset after tuning last layers
learner.unfreeze()
learner.lr_find()                               ## find suitable lr range for training all the layers NOW
learner.recorder.plot()
learner.fit_one_cycle(cyc_len=epochs, max_lr=slice(1e-6, 1e-3))     ## using the same earlier lr range to train all the layers now
                                                                    ## max_lr range here is obtained after analyzing the lr_find plot

learner.recorder.plot_losses()
learner.recorder.plot_metrics()
learner.save("stage-2")                         ## SAVING THE TRAINED MODEL

####################################################################################################
