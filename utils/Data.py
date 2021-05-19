import pandas as pd
import pickle
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from utils.tta_augmentation import *


def split_train_valid_folds(BaseDatasetClass, fold_file, transform, kwargs):
    data_dir = kwargs["data_dir"]

    return_fold_dataset = dict()
    fold_csv = pd.read_csv(fold_file)
    for fold_name in fold_csv.columns.tolist():
        valid_subjects = fold_csv[fold_name].tolist()
        if len(valid_subjects) == 0:
            train_dataset = BaseDatasetClass(phase="train",
                                             data_dir=data_dir,
                                             valid_subjects=valid_subjects,
                                             plaque_file=kwargs["plaque_file"],
                                             polarize=kwargs["polarize"],
                                             valid=False,
                                             transform=transform)
            valid_dataset = None
        else:
            train_dataset = BaseDatasetClass(phase="train",
                                             data_dir=data_dir,
                                             valid_subjects=valid_subjects,
                                             plaque_file=kwargs["plaque_file"],
                                             valid=False,
                                             transform=transform)
            valid_dataset = BaseDatasetClass(phase="train",
                                             data_dir=data_dir,
                                             valid_subjects=valid_subjects,
                                             plaque_file=kwargs["plaque_file"],
                                             valid=True,
                                             transform=None)
        return_fold_dataset[fold_name] = dict(
            train=train_dataset,
            valid=valid_dataset
        )

    return return_fold_dataset


def callback_get_label(dataset, idx):
    # callback function used in imbalanced dataset loader.
    target = dataset[idx]['label']
    return target


def get_sample_weight(dataset):
    labels = []
    for idx in range(len(dataset)):
        label = dataset[idx]['label']
        labels.append(int(label))
    labels = np.array(labels)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    return samples_weight


def get_weighted_sampler(dataset):
    """
    :Example:
     >>> DataLoader(dataset,
     >>>            batch_size=32,
     >>>            sampler=get_weighted_sampler(dataset),
     >>>            shuffle=False)

     When using this sampler, set shuffle as False
    """
    samples_weight = get_sample_weight(dataset)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def get_dataloader(dataset, sampler, kwargs):
    return DataLoader(dataset,
                      batch_size=kwargs["batch_size"],
                      sampler=sampler,
                      shuffle=kwargs["shuffle"])


def get_dataloader_from_folds(dataset_folds, kwargs):
    return_fold_dataloader = dict()
    for fold_name in dataset_folds.keys():
        print(f"{fold_name} / {len(dataset_folds)} Dataloader Proceeding ...")
        train_dataset = dataset_folds[fold_name]['train']
        valid_dataset = dataset_folds[fold_name]['valid']
        return_fold_dataloader[fold_name] = dict(
            train=get_dataloader(train_dataset, get_weighted_sampler(train_dataset), kwargs),
            valid=get_dataloader(valid_dataset, None, kwargs) if valid_dataset is not None else None
        )
        print(f"{fold_name} fold Ready")
    print(f"\n Folded Dataloader Ready")
    return return_fold_dataloader


def normalize(img):
    _mean = img.mean()
    _std = img.std()
    return (img - _mean) / _std


if __name__ == '__main__':
    pass
