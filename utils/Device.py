import torch
from torch.nn import DataParallel
import os


def gpu_setting(gpus: str = "0123"):
    if len(gpus) > 1:
        gpus = ", ".join([x for x in gpus])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
        multi_gpu = True
    elif len(gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
        multi_gpu = False
    else:
        multi_gpu = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available device: {device}")

    if device == "cuda":
        print(f"Number of Cuda device: {torch.cuda.device_count()}")
    elif device == "cpu":
        print(f"Number of Cuda device: None, CPU instead")

    return device, multi_gpu


def model_dataparallel(model, multi_gpu, device):
    model = model.to(device)
    if multi_gpu:
        model = DataParallel(model)
    print(f"Model Loaded, Multi GPU: {multi_gpu}")
    return model


if __name__ == '__main__':
    pass
