from models.Base import *
from utils.Device import model_dataparallel, gpu_setting

import torch.optim as optim


def load_model(args, phase="train"):

    model_dict = dict(
        densenet=BaseNet,
    )

    model = model_dict[args["model_name"]](**args["model_param"])

    # gpu setting
    device, multi_gpu = gpu_setting(args["gpus"])
    print(device, multi_gpu)

    if phase == "test":
        model.eval()

    if args["dp"]:
        model = model_dataparallel(model, multi_gpu, device)

    if args["model_path"] is not None:
        model_state_dict = torch.load(args["model_path"])[args["state_dict_key"]]
        model.load_state_dict(model_state_dict)

    return model


def load_optimizer(args):
    query_optimizer = args.get("optimizer").lower()
    optim_dict = dict(
        adam=optim.Adam,
        sgd=optim.SGD
    )
    return optim_dict.get(query_optimizer)


if __name__ == '__main__':
    pass
