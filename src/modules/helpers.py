"""
Helper functions to do data cleaning, preprocessing and postprocessing
"""

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import pathlib
import numpy as np

__author__ = "Vinay Kumar"
__copyright__ = "@vinaykumar2491, 2019"
__project__ = "flixstockJam"

def remove_ghost_data(df:pd.DataFrame=None, data_path:pathlib.Path=None, inplace:bool=True, col=None):
    for x in df[col]:
        if not (data_path/x).exists():
            df.drop(df[df[col] == x].index, inplace=inplace)
    if inplace:
        return None
    else:
        return df

def preprocess_from_csv(data_path: pathlib.Path=None, attr_path: pathlib.Path=None,
                        data_augmentation: bool=False, multilabel_classification: bool=False,
                        target_attr: str=None, multilabel_start: int=1, multilabel_end: int=4,
                        multilabel_delim: str=" : ") -> pd.DataFrame:
    ## Step-1: Load and analyze the database
    df = pd.read_csv(attr_path)
    print(f"{df.head()}")               ## viewing and analyzing the provided data/labels & its structure

    ## Step-2: PREPROCESSING the database from csv
    ## After looking at the dataframe, I am restructuring the data representaion to convenience as:
    ## 2.a. Check for any NaN in the labels (neck, ...).
    ## 2.b. Create a separate label for each item representing a boolean if the item has NaN for the
    ##      respective labels. Create separate new labels for each existing label types (eg. neck_NaN).
    ## 2.c. Replace the NaN values in the original label types with the mean of that label type in DB.
    ## 2.d. Concat all the original labels for an item and save as a new label called "attr". We will be
    ##      using this "attr" as a label for the respective image item to train our neuralNet.
    ## 2.e. Creating the dataset which can be used by the learner

    orig_new_ref = {}

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

    if multilabel_classification:
        ## Step 2.d
        df[target_attr] = ""
        for i in df.columns[multilabel_start:multilabel_end]:
            if df[target_attr].all() == "":
                df[target_attr] = df[i]
            else:
                df[target_attr] = df[target_attr] + multilabel_delim + df[i]

    ## Sanity check: remove the files from csv that donot exist in images
    img_col = df.columns[0]
    # df.drop(df[df.filename=="11495010560702-The-Indian-Garage-Co-Men-Tshirts-7851495010560569-2.jpg"].index, inplace=True)
    remove_ghost_data(df, data_path, inplace=True, col=img_col)
    all_label_types = np.array([])
    if multilabel_classification:
        for i in df.columns[multilabel_start:multilabel_end]:
            all_label_types = np.concatenate((all_label_types, df[i].unique()))
        all_label_types = sorted(all_label_types)

    return (df, all_label_types)


def create_embeddings(label: str, all_label_types, delim: str):
    label = label.split(delim)
    label_embedding = np.zeros(all_label_types.size, dtype=int)
    for i in label:
        label_embedding[np.where(all_label_types==i)] = 1


class FlixDataset(torch.utils.data.Dataset):
    """
    Custom dataset for flexNet task.
    """

    def __init__(self, df, data_path, in_col, target_col, device, all_label_types, delim):
        """
        Inputs:
        - df          [type=str] : path of the csv file which contains the information about the data
        - root_dir    [type=str] : path of the directory with all input data
        """

        self.df = df
        self.data_path = data_path
        self.in_col = in_col
        self.target_col = target_col
        self.device = device
        self.all_label_types = all_label_types
        self.delim = delim

    def __len__(self):
        return len(self.df[in_col])

    def __getitem__(self, idx):
        input_img = Image.open(self.data_path/self.df[self.in_col][idx])
        label = create_embeddings(label=self.df[self.target_col][idx], all_label_types=self.all_label_types, delim=self.delim)
        input_tensor = TF.to_tensor(input_img).to(self.device)
        label_tensor = torch.from_numpy(label).to(self.device)
        if idx == 0:
            print(input_tensor.shape, label_tensor.shape, label_tensor)
        return {"input": input_tensor, "label": label_tensor}
