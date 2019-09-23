"""
Helper functions to do data cleaning, preprocessing and postprocessing
"""

import torch
import pandas as pd
import pathlib

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
    return df

