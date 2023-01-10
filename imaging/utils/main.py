import torch
from .ct_loader import CTLoader, CTLoader2D
from .trainer import Trainer

def set_home(device: int):
    if torch.cuda.is_available():
        dir = "/media/avcstorage/gravo"
        torch.cuda.set_device(device)
    else:
        dir = "../../../data/gravo"
    return dir
    
    
def train(to_test: list, N: int, trainer):
    for model in to_test:
        for i in range(N):
            trainer.train(model)
    print("done")


def main(to_test: list, N: int = 3, device: int = 3, slice: str = None, 
    slice_range: int = None, pad: int = None, batch_size = 32):
    dir = set_home(device)
    if slice_range is not None:
        assert slice is not None
        ct_loader = CTLoader2D(slice, slice_range = slice_range, data_dir = dir, pad = pad)
        for model in to_test:
            model.set_slice_info(slice_range, slice)
    else:
        ct_loader = CTLoader(data_dir = dir)
    trainer = Trainer(ct_loader, batch_size = batch_size)
    train(to_test, N, trainer)
    
