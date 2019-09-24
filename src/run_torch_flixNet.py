"""
Raw implementation of training and validation script using pytorch for the task of flixstock Jam
"""

import gc
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import pathlib

from modules import helpers
from modules import flixNet

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
model_dir = pathlib.Path(f"../../trained_models")

## Hyperparams
train_split = 0.7
valid_split = 0.2
test_split = 0.1
test_ds = None
bs = 2
epochs = 5
torch.manual_seed(217)          ## setting seed for Pytorch RNG

optim_hyperparams = {'adam':{'beta_1':0.9,
                            'beta_2':0.999,
                            'epsilon':1e-08,
                            'weight_decay':0.004
                            }
                    }
loss_history = {}
learning_rates = []
for itr in range(10):
    lr = 10**np.random.uniform(-3,-6)
    learning_rates.append(lr)

df = pd.DataFrame()
if data_preprocessing:
    print(f"##-----------------\nPreprocesssing starting...\n##-----------------")
    df, all_label_types = helpers.preprocess_from_csv(data_path=data_path, attr_path=attr_csv_path,
                                                      data_augmentation=False, multilabel_classification=True,
                                                      target_attr="attr", multilabel_start=1,
                                                      multilabel_end=4, multilabel_delim=" : ")
    print(all_label_types)
    print(df.head())
    print(f"##-----------------\nPreprocesssing done.\n##-----------------")

if do_training:
    ## Step-1: Preparing the training and validation dataset using Pytorch
    dataset = helpers.FlixDataset(df=df, data_path=data_path, in_col="filename", target_col="attr",
                                  device=device, all_label_types=all_label_types, delim=" : ")
    train_ds_len = round(len(dataset)*train_split)
    valid_ds_len = round(len(dataset)*valid_split)
    test_ds_len = len(dataset) - train_ds_len - valid_ds_len
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(dataset=dataset,
                                                                lengths=[train_ds_len, valid_ds_len, test_ds_len])
    # Creating the dataloader
    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)

    ## Step-2: Preparing the flixNet architecture
    ## Step-3: Designing the training and validation hyperparams, metrics etc
    ## Step-4: Training and validating the flixNet

    for lr in learning_rates:
        flixnet = flixNet.FlixNet()
        flixnet.apply(helpers.weights_init)         ## initializing the parameters of flixnet
        # initialize wights if using an untrained arch otherwise not
        if torch.cuda.is_available():
            flixnet.cuda()

        optimizer = torch.optim.Adam(params=flixnet.parameters(), lr=lr,
                                    betas=(optim_hyperparams['adam']['beta_1'], optim_hyperparams['adam']['beta_2']),
                                    eps=optim_hyperparams['adam']['epsilon'],
                                    weight_decay=optim_hyperparams['adam']['weight_decay'], amsgrad=True)
        loss_criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

        loss_history[lr] = []
        for epoch in range(epochs):
            run_loss = 0.0
            count = 0
            for _, sampled_batch in enumerate(train_dl):
                input_batch = sampled_batch["input"]
                input_batch = input_batch.float().view(-1,3,300,225)
                # print(input_batch.shape)
                label_batch = sampled_batch["label"]
                label_batch = label_batch.float().view(-1,21)

                if torch.cuda.is_available():
                    input_batch = input_batch.cuda()
                    label_batch = label_batch.cuda()
                optimizer.zero_grad()
                pred_batch = flixnet(input_batch)
                print(pred_batch.shape, label_batch)
                loss = loss_criterion(pred_batch, label_batch)
                run_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                count += input_batch.shape[0]
            loss_history[lr].append(run_loss/count)
            print(f"lr={lr} : [{epoch}/{epochs}] : {loss_history[lr][-1]}")

        ## Step-6: Saving the model, so as to be used during testing.
        torch.save(flixnet, model_dir/f"{lr}.model")
        print(f"Trained model saved at {model_dir}/{lr}.model")




