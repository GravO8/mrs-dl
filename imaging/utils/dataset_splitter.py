import os
import pandas as pd
import numpy as np

DATA_DIR        = "../../../data/gravo"
TEST_SIZE       = 60
VAL_PERC        = .1
DATASET_SIZE    = 465
SET             = "set"
PATIENT_ID      = "patient_id"
RANKIN          = "rankin"
BINARY_RANKIN   = "binary_rankin"
AUGMENTATION    = "augmentation"
AUGMENTS        = ["RandomFlip", "RandomNoise", "RandomElasticDeformation", "RandomAffine"]


def get_patients(table_data: pd.DataFrame, nccts: list):
    patients = {PATIENT_ID:[], RANKIN:[], BINARY_RANKIN: []}
    for _, row in table_data.iterrows():
        patient_id  = row["idProcessoLocal"]
        rankin      = row["rankin-23"]
        if rankin.isnumeric() and (row["NCCT"] == "OK"):
            assert f"{patient_id}.nii" in nccts, f"get_patients: {patient_id} doesn't have a NCCT scan."
            rankin = int(rankin)
            patients[PATIENT_ID].append(patient_id)
            patients[RANKIN].append(rankin)
            patients[BINARY_RANKIN].append(int(rankin > 2))
    return pd.DataFrame(patients)
    
    
def dataset_split(dataset: pd.DataFrame):
    '''
    +-------------------------------+-----+------+
    |             train             | val | test |
    +-------------------------------+-----+------+
    is equivalent to
    +-------------------------------------+------+
    |              development            | test |
    +-------------------------------------+------+
    '''
    assert len(dataset) == DATASET_SIZE, "dataset_split: unexpected dataset size"
    dataset    = dataset.sample(frac = 1, random_state = 0).reset_index(drop = True)
    N0, N1     = np.bincount(dataset[BINARY_RANKIN].values) # N1 = number of examples of class 1
    N1_TEST    = int(N1*TEST_SIZE/DATASET_SIZE)             # N1_TEST = number of examples of class 1 in the test set
    N0_TEST    = TEST_SIZE - N1_TEST
    N1_DEV     = N1 - N1_TEST                               # N1_DEV = number of examples of class 1 in the development set
    N0_DEV     = N0 - N0_TEST
    N1_VAL     = int(VAL_PERC*N1_DEV)                       # N1_VAL = number of examples of class 1 in the validation set
    N0_VAL     = int(VAL_PERC*N0_DEV)
    sets       = []
    n1_test    = 0
    n0_test    = 0
    n1_val     = 0
    n0_val     = 0
    for _, row in dataset.iterrows():
        if row[BINARY_RANKIN] == 1:
            if n1_test < N1_TEST:
                sets.append("test")
                n1_test += 1
            elif n1_val < N1_VAL:
                sets.append("val")
                n1_val += 1
            else:
                sets.append("train")
        else:
            if n0_test < N0_TEST:
                sets.append("test")
                n0_test += 1
            elif n0_val < N0_VAL:
                sets.append("val")
                n0_val += 1
            else:
                sets.append("train")
    dataset[SET] = sets
    assert TEST_SIZE == len(dataset[dataset[SET] == "test"])
    assert N1_TEST == len(dataset[(dataset[SET] == "test") & (dataset[BINARY_RANKIN] == 1)])
    assert N0_TEST == len(dataset[(dataset[SET] == "test") & (dataset[BINARY_RANKIN] == 0)])
    assert N1_VAL == len(dataset[(dataset[SET] == "val") & (dataset[BINARY_RANKIN] == 1)])
    assert N0_VAL == len(dataset[(dataset[SET] == "val") & (dataset[BINARY_RANKIN] == 0)])
    assert (N1-N1_TEST-N1_VAL) == len(dataset[(dataset[SET] == "train") & (dataset[BINARY_RANKIN] == 1)])
    assert (N0-N0_TEST-N0_VAL) == len(dataset[(dataset[SET] == "train") & (dataset[BINARY_RANKIN] == 0)])
    return dataset
    
    
def prettify_dataset(dataset: pd.DataFrame):
    sort_col        = [("test", "val", "train").index(row[SET]) for _, row in dataset.iterrows()]
    dataset["sort"] = sort_col
    dataset         = dataset.sort_values(by = ["sort", RANKIN])
    del dataset["sort"]
    return dataset
    
    
def get_augmentations(dataset: pd.DataFrame):
    augmentations   = {PATIENT_ID:[], AUGMENTATION:[]}
    n_augments      = len(AUGMENTS)                                 # number of available augmentations
    train_set       = dataset[dataset[SET] == "train"]
    n_train         = np.bincount(train_set[BINARY_RANKIN].values)
    n_augment       = n_train[1]*(n_augments+1)                     # the desired number of examples in both classes, after augmentations (1 is the minority class)
    for label in (0, 1):
        ids                 = train_set[train_set[BINARY_RANKIN] == label][PATIENT_ID].values
        to_add              = n_augment - n_train[label]
        transforms_to_add   = [to_add // n_augments] * n_augments
        for i in range(to_add % n_augments):
            transforms_to_add[i] += 1
        for i in range(n_augments):
            for id in np.random.choice(ids, size = transforms_to_add[i], replace = False):
                augmentations[PATIENT_ID].append(id)
                augmentations[AUGMENTATION].append(AUGMENTS[i])
    augmentations = pd.DataFrame(augmentations)
    augmentations = train_set.merge(augmentations, how = "left")[[PATIENT_ID,AUGMENTATION]]
    augmentations.dropna(inplace = True)
    assert len(train_set)+len(augmentations) == n_augment*2
    # assert the total number of examples in each class after augmentations is the same (and equal to n_augment)
    assert n_train[0]+len(augmentations[augmentations[PATIENT_ID].isin(train_set[train_set[BINARY_RANKIN] == 0][PATIENT_ID].values)]) == n_augment
    assert n_train[1]+len(augmentations[augmentations[PATIENT_ID].isin(train_set[train_set[BINARY_RANKIN] == 1][PATIENT_ID].values)]) == n_augment
    return augmentations


if __name__ == "__main__":
    table_data_dir  = os.path.join(DATA_DIR, "table_data.csv")
    nccts_dir       = os.path.join(DATA_DIR, "NCCT")
    table_data      = pd.read_csv(table_data_dir)
    nccts           = [file for file in os.listdir(nccts_dir) if file.endswith(".nii")]
    
    dataset         = get_patients(table_data, nccts)
    dataset         = dataset_split(dataset)
    dataset         = prettify_dataset(dataset)
    
    augmentations   = get_augmentations(dataset)
    
    augmentations.to_csv("augmentations.csv", index = False)
    dataset.to_csv("dataset.csv", index = False)
