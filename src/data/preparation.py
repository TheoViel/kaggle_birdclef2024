import os
import re
import pandas as pd

from itertools import chain
from sklearn.model_selection import StratifiedGroupKFold


COLS = [
    "id",
    "filename",
    "primary_label",
    "secondary_labels",
    "rating",
    "path",
    "path_ft",
    "fold",
]

TO_REMOVE = [  # Duplicates - Different ID, same class
    'asbfly/XC724148.ogg', 'barswa/XC575747.ogg', 'bcnher/XC669542.ogg', 'bkskit1/XC350249.ogg',
    'blhori1/XC417133.ogg', 'blhori1/XC537503.ogg', 'blrwar1/XC662285.ogg', 'brakit1/XC537471.ogg',
    'brcful1/XC157971.ogg', 'brnshr/XC510750.ogg', 'btbeat1/XC513403.ogg', 'btbeat1/XC683300.ogg',
    'btbeat1/XC743619.ogg', 'categr/XC438523.ogg', 'cohcuc1/XC241127.ogg', 'cohcuc1/XC423419.ogg',
    'comgre/XC175341.ogg', 'comgre/XC192404.ogg', 'comgre/XC58586.ogg', 'comior1/XC303819.ogg',
    'comkin1/XC690633.ogg', 'commyn/XC577886.ogg', 'commyn/XC652901.ogg', 'compea/XC644022.ogg',
    'comsan/XC385908.ogg', 'comsan/XC642698.ogg', 'comsan/XC667806.ogg', 'comtai1/XC122978.ogg',
    'comtai1/XC304811.ogg', 'comtai1/XC540351.ogg', 'comtai1/XC540352.ogg', 'crfbar1/XC615778.ogg',
    'dafbab1/XC157972.ogg', 'dafbab1/XC187059.ogg', 'dafbab1/XC187068.ogg', 'dafbab1/XC187069.ogg',
    'eaywag1/XC527598.ogg', 'eucdov/XC124694.ogg', 'eucdov/XC347428.ogg', 'eucdov/XC355152.ogg',
    'eucdov/XC368596.ogg', 'eucdov/XC747408.ogg', 'eucdov/XC788267.ogg', 'graher1/XC357551.ogg',
    'graher1/XC590144.ogg', 'grbeat1/XC303999.ogg', 'grecou1/XC365425.ogg', 'grewar3/XC537475.ogg',
    'grnwar1/XC157973.ogg', 'grtdro1/XC613192.ogg', 'grywag/XC457124.ogg', 'grywag/XC575901.ogg',
    'grywag/XC592019.ogg', 'grywag/XC655063.ogg', 'grywag/XC745650.ogg', 'grywag/XC812495.ogg',
    'heswoo1/XC357149.ogg', 'heswoo1/XC665715.ogg', 'hoopoe/XC252584.ogg', 'hoopoe/XC365530.ogg',
    'houcro1/XC683047.ogg', 'houspa/XC326674.ogg', 'inbrob1/XC744706.ogg', 'insowl1/XC301142.ogg',
    'junbab2/XC282586.ogg', 'labcro1/XC19736.ogg', 'labcro1/XC265731.ogg', 'labcro1/XC312582.ogg',
    'laudov1/XC405374.ogg', 'lblwar1/XC157974.ogg', 'lewduc1/XC254813.ogg', 'litegr/XC447850.ogg',
    'litegr/XC448898.ogg', 'litgre1/XC630560.ogg', 'litgre1/XC663244.ogg', 'litspi1/XC721636.ogg',
    'litspi1/XC721637.ogg', 'litswi1/XC440301.ogg', 'lobsun2/XC157975.ogg', 'maghor2/XC157978.ogg',
    'maghor2/XC786587.ogg', 'malpar1/XC157976.ogg', 'marsan/XC383288.ogg', 'marsan/XC716673.ogg',
    'mawthr1/XC455211.ogg', 'orihob2/XC557293.ogg', 'piebus1/XC122395.ogg', 'piebus1/XC792272.ogg',
    'placuc3/XC486683.ogg', 'placuc3/XC572950.ogg', 'plaflo1/XC614946.ogg', 'purher1/XC827207.ogg',
    'pursun4/XC514852.ogg', 'rewlap1/XC732874.ogg', 'rorpar/XC199339.ogg', 'rorpar/XC516402.ogg',
    'sohmyn1/XC743682.ogg', 'spepic1/XC804431.ogg', 'spoowl1/XC591177.ogg', 'stbkin1/XC199815.ogg',
    'stbkin1/XC266682.ogg', 'stbkin1/XC406138.ogg', 'vefnut1/XC157979.ogg', 'vefnut1/XC289785.ogg',
    'wemhar1/XC590354.ogg', 'whbbul2/XC335670.ogg', 'whbsho3/XC856463.ogg', 'whbsho3/XC856468.ogg',
    'whbwat1/XC840071.ogg', 'whiter2/XC265267.ogg', 'whtkin2/XC157981.ogg', 'whtkin2/XC430256.ogg',
    'whtkin2/XC540087.ogg', 'woosan/XC476064.ogg', 'woosan/XC578599.ogg', 'woosan/XC740798.ogg',
    'zitcis1/XC302781.ogg',
]

RENAME_DUPS = {  # Different id, different class
    'XC207123': 'XC207062',
    'XC268375': 'XC241382',
    'XC823514': 'XC823527',
    'XC713308': 'XC713467',
    'XC825766': 'XC825765',
    'XC402325': 'XC402326',
    'XC247286': 'XC197438',
    'XC185511': 'XC185505',
    'XC163930': 'XC163901',
}


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
    """
    Updates secondary labels by combining labels for entries with duplicate IDs.

    Args:
        df (pandas.DataFrame): Data, expects columns 'id', 'primary_label', and 'secondary_labels'.

    Returns:
        pandas DataFrame: Updated data.
    """
    df_dups = df[df.duplicated(subset="id", keep=False)].groupby("id").agg(list)

    if not len(df_dups):
        return df

    df_dups["labels"] = df_dups.apply(
        lambda x: x.primary_label + list(chain.from_iterable(x.secondary_labels)),
        axis=1,
    )
    mapping = df_dups.to_dict()["labels"]
    df["secondary_labels"] = df.apply(
        lambda x: mapping.get(x.id, x.secondary_labels), axis=1
    )
    df["secondary_labels"] = df.apply(
        lambda x: list(set([s for s in x.secondary_labels if s != x.primary_label])),
        axis=1,
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


def update_labels(df):
    """
    Update primary and secondary labels in a DataFrame
    by combining labels for entries with duplicate IDs.

    Args:
        df (pandas.DataFrame): Data, expects columns 'id', 'primary_label', and 'secondary_labels'.

    Returns:
        pandas.DataFrame: DataFrame with updated 'primary_label' and 'secondary_labels'.
    """
    df_dups = df[df.duplicated(subset="id", keep=False)].groupby("id").agg(list)

    if not len(df_dups):
        return df

    df_dups["secondary_labels"] = df_dups["secondary_labels"].apply(
        lambda x: list(chain.from_iterable(x))
    )
    mapping_1 = df_dups.to_dict()["primary_label"]
    mapping_2 = df_dups.to_dict()["secondary_labels"]
    df["primary_label"] = df.apply(
        lambda x: mapping_1.get(x.id, x.primary_label), axis=1
    )
    df["secondary_labels"] = df.apply(
        lambda x: mapping_2.get(x.id, x.secondary_labels), axis=1
    )
    df["secondary_labels"] = df.apply(
        lambda x: list(set([s for s in x.secondary_labels if s not in x.primary_label])),
        axis=1,
    )
    return df


def prepare_data_2(data_path="../input/"):
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
    df["path_ft"] = (
        data_path + "train_features/" + df["filename"].apply(lambda x: x[:-3]) + "hdf5"
    )
    df["secondary_labels"] = df["secondary_labels"].apply(eval)

    # Handle duplicates
    df = df[~df["filename"].isin(TO_REMOVE)].reset_index(drop=True)
    df["id"] = df["id"].apply(lambda x: RENAME_DUPS.get(x, x))

    df = update_labels(df)
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

    folds = pd.read_csv(data_path + "folds_4.csv")
    df = df.merge(folds)

    return df


def prepare_xenocanto_data(data_path="../input/"):
    """
    Prepares the xenocanto data.

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
        data_path
        + "xenocanto/features/"
        + df["filename"].apply(lambda x: x[:-3])
        + "hdf5"
    )
    df["secondary_labels"] = df["also"]
    rating_mapping = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0, "no score": 3.75}
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
    df = df[
        ~df["id"].isin(dups)
    ]  # This could be handled better using update_secondary_labels fct
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


def add_xeno_low_freq(df, low_freq=500, remove_low_rating=False, verbose=0):
    """
    Add samples from Xenocanto dataset to enrich classes with low frequency.

    Args:
        df (pandas.DataFrame): DataFrame containing the main dataset.
        low_freq (int, optional): Threshold frequency for low-frequency classes. Defaults to 500.
        remove_low_rating (bool, optional): Remove samples with low ratings. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        pandas.DataFrame: DataFrame containing additional samples from Xenocanto dataset.
    """
    df_xc = prepare_xenocanto_data()

    df_xc = df_xc[~df_xc["id"].isin(df["id"])]
    if remove_low_rating:
        df_xc = df_xc[df_xc["rating"] != 1]

    df_ = df[~df["primary_label"].apply(lambda x: isinstance(x, list))]
    cts = df_.groupby('primary_label').count()["id"]
    cts = cts[cts < low_freq]

    extra_samples = []
    for c, v in pd.DataFrame(cts).iterrows():
        xc_samples = df_xc[df_xc["primary_label"] == c]
        xc_samples = xc_samples.head(low_freq - v.id)
        extra_samples.append(xc_samples)
        if verbose:
            print(f'Add {len(xc_samples)} {c} samples')

    return pd.concat(extra_samples)


def upsample_low_freq(df, low_freq=20, verbose=0):
    """
    Upsamples low-frequency classes in the dataset.

    Args:
        df (pandas.DataFrame): DataFrame containing the dataset.
        low_freq (int, optional): Threshold frequency for low-frequency classes. Defaults to 20.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        pandas.DataFrame: DataFrame containing the upsampled dataset.
    """
    df_ = df[~df["primary_label"].apply(lambda x: isinstance(x, list))]
    cts = df_.groupby('primary_label').count()["id"]
    cts = cts[cts < low_freq]

    extra_samples = []
    for c, v in pd.DataFrame(cts).iterrows():
        if v.id > low_freq:
            continue

        samples = df[df["primary_label"] == c]
        n = 20 // len(samples)
        for _ in range(n):
            extra_samples.append(samples)
        if verbose:
            print(f'Duplicate {len(samples)} {c} samples {n} times')

    return pd.concat(extra_samples)
