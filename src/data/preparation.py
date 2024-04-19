import os
import re
import glob
import pandas as pd

from itertools import chain
from sklearn.model_selection import StratifiedGroupKFold


COLS = ["id", "filename", "primary_label", "secondary_labels", "rating", "path", "path_ft", "fold"]


def prepare_folds(data_path="../input/", k=4):
    """
    Prepare data folds for cross-validation.
    StratifiedGroupKFold is used.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".
        k (int, optional): Number of cross-validation folds. Defaults to 4.

    Returns:
        pandas DataFrame: DataFrame containing the files and their respective fold assignments.
    """
    df = prepare_data(data_path)

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    splits = sgkf.split(df, y=df["primary_label"], groups=df["id"])

    df["fold"] = -1
    for i, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = i

    df_folds = df[["filename", "fold"]]
    df_folds.to_csv(data_path + f"folds_{k}.csv", index=False)
    return df_folds


def update_secondary_labels(df):
    df_dups = df[df.duplicated(subset="id", keep=False)].groupby('id').agg(list)

    if not len(df_dups):
        return df

    df_dups["labels"] = df_dups[["filename", "primary_label", "secondary_labels"]].apply(
        lambda x: x.primary_label + list(chain.from_iterable(x.secondary_labels)), axis=1
    )
    mapping = df_dups.to_dict()['labels']
    df["secondary_labels"] = df.apply(lambda x: mapping.get(x.id, x.secondary_labels), axis=1)
    df["secondary_labels"] = df.apply(
        lambda x: list(set([s for s in x.secondary_labels if s != x.primary_label])), axis=1
    )
    return df


def prepare_data(data_path="../input/"):
    """
    Prepares the data.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".

    Returns:
        pandas DataFrame: Metadata
    """
    df = pd.read_csv(os.path.join(data_path, "train_metadata.csv"))
    df["id"] = df["filename"].apply(lambda x: x.split("/")[-1][:-4])
    df = df[["id", "filename", "primary_label", "secondary_labels", "rating"]]
    df["path"] = data_path + "train_audio/" + df["filename"]
    df["path_ft"] = data_path + "train_features/" + df["filename"].apply(lambda x: x[:-3]) + "hdf5"

    df["secondary_labels"] = df["secondary_labels"].apply(eval)
    df = update_secondary_labels(df)

    folds = pd.read_csv(data_path + "folds_4.csv")
    df = df.merge(folds)

    return df


def prepare_xenocanto_data(data_path="../input/"):
    """
    Prepares the data.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".

    Returns:
        pandas DataFrame: Metadata
    """
    df = pd.read_csv(
        data_path + "xenocanto/BirdClef2024_additional.csv", low_memory=False
    )
    df = df[["id", "en", "file", "primary_label", "also", "q"]]

    df["id"] = "XC" + df["id"].astype(str)
    df["filename"] = df["primary_label"] + "/" + df["file"] + ".mp3"
    df["path"] = data_path + "xenocanto/audio/" + df["filename"]
    df["path_ft"] = (
        data_path + "xenocanto/features/" + df["filename"].apply(lambda x: x[:-3]) + "hdf5"
    )
    df["secondary_labels"] = df["also"]
    rating_mapping = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0, 'no score': 3.75}
    df["rating"] = df["q"].map(rating_mapping).fillna(3.75)
    df["fold"] = -1

    # Remove files already in comp data and not found
    dups = [
        "XC184468",
        "XC239509",
        "XC371015",
        "XC445303",
        "XC447036",
        "XC460945",
        "XC467373",
        "XC503389",
        "XC514027",
        "XC522123",
        "XC581045",
        "XC589069",
        "XC724832",
    ]
    df = df[~df["id"].isin(dups)]  # This could be handled better using update_secondary_labels fct
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    # Secondary labels & mapping
    df["secondary_labels"] = df["secondary_labels"].apply(eval)

    df_map = pd.read_csv(data_path + "eBird_Taxonomy_v2021.csv")
    mapping = {}
    for col in ["SCI_NAME", "PRIMARY_COM_NAME"]:
        df_map[col] = df_map[col].apply(lambda x: re.sub(r"\s+", "", x.lower()))
        mapping.update(df_map.set_index(col).to_dict()["SPECIES_CODE"])
    df["secondary_labels"] = df["secondary_labels"].apply(
        lambda x: [mapping.get(re.sub(r"\s+", "", k.lower()), "unk") for k in x]
    )
    df["secondary_labels"] = df["secondary_labels"].apply(
        lambda x: [k for k in x if k != "unk"]
    )
    df = update_secondary_labels(df)

    return df[COLS]


def prepare_nocall_data(data_path="../input/"):
    """
    Prepares the data.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".

    Returns:
        pandas DataFrame: Metadata
    """
    df = pd.DataFrame({"path_ft": glob.glob(data_path + "nocall_features/*/*.hdf5")})
    df["path"] = df["path_ft"]
    df["id"] = df["path_ft"].apply(lambda x: x.split("/")[-1][:-5])
    df["filename"] = df["id"]
    df["primary_label"] = "nocall"
    df["secondary_labels"] = [[] for _ in range(len(df))]
    df["rating"] = 2.5
    df["fold"] = -1

    return df[COLS]
