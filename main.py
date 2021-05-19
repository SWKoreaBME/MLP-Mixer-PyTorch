from runners.Trainer import trainer
from runners.Tester import tester
from scripts.config import args
from loader.Base import IvusDataset
from utils.Logger import Logger
from utils.Metric import ModelMetrics
from utils.Augmentation import Transform
from utils.Device import *
from utils.Data import *
from utils.Version import args_save_to_txt
from utils.Model import load_model, load_optimizer

import argparse
import os
import torch.optim as optim
import torch.nn as nn


def main():
    # Show all running arguments
    for key, value in args.items():
        print(f"{key}: {value}")

    # save argument parse arguments
    os.makedirs(args["save_dir"], exist_ok=True)

    # Save version log
    args_save_to_txt(os.path.join(args["save_dir"], "args.txt"), args)

    # # # # #
    # Train #
    # # # # #

    if phase == "train":

        # Dataloader
        transform = Transform(phase).data_augmentation

        assert args["fold_file"] is not None, "No Fold file!"
        dataset_folds = split_train_valid_folds(IvusDataset, args["fold_file"], transform, args)
        dataloader_folds = get_dataloader_from_folds(dataset_folds, args)

        for fold_name in dataloader_folds.keys():
            # load model and wrap with dataparallel
            # Wrap these code lines with a function called load_model

            model = load_model(args)

            optimizer = load_optimizer(args)(model.parameters(), lr=args["learning_rate"])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args["step_size"])
            loss_function = nn.BCELoss()

            print(f"\nProceeding {fold_name} Training ...")

            fold_save_dir = os.path.join(args["save_dir"], fold_name)
            os.makedirs(fold_save_dir, exist_ok=True)

            logger = Logger(save_dir=fold_save_dir, total_epoch=args["epochs"])

            train_dataloader = dataloader_folds[fold_name]["train"]
            valid_dataloader = dataloader_folds[fold_name]["valid"]

            print(f"Train: {len(train_dataloader.dataset)}")
            if valid_dataloader is not None:
                print(f"Valid: {len(valid_dataloader.dataset)}")

            run_args = dict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_function=loss_function,
                logger=logger,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                **args
            )

            trainer(run_args)

    # # # # #
    # Test  #
    # # # # #

    elif phase == "test":

        model = load_model(args, phase)
        test_dataset = IvusDataset(phase,
                                   transform=None,
                                   polarize=args.get("polarize"),
                                   plaque_file=args.get("plaque_file"))  # 임시로 phase 를 train 으로 만들어서 inference
        test_dataloader = get_dataloader(test_dataset, None, args)

        run_args = dict(
            model=model,
            test_dataloader=test_dataloader,
            metrics=ModelMetrics,
            **args
        )
        tester(run_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Project Description')
    parser.add_argument('--test', action='store_true', dest='test', help='Test module')
    p_args = parser.parse_args()

    phase = "test" if p_args.test else "train"
    args = args[phase]
    main()
