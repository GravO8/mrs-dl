import torch
from abc import ABC, abstractmethod
from .models import final_mlp
from .encoder import Encoder
        
        
class Model(ABC, torch.nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder    = encoder
        self.mlp        = final_mlp(self.encoder.out_channels)
        
    @abstractmethod
    def process_input(self, x):
        pass
        
    @abstractmethod
    def name_appendix():
        pass
        
    def normalize_input(self, x, range_max: int = 100):
        return x/range_max
    
    def forward(self, x):
        x = self.process_input(x)
        x = self.encoder(x)
        return self.mlp(x)
        
    def get_name(self):
        return self.name_appendix() + "-" + self.encoder.get_name()


class Baseline3DCNN(Model):
    def process_input(self, x):
        return self.normalize_input(x)
    def name_appendix(self):
        return "baseline-3DCNN"
        
        
class BaselineMirror(Model):
    def process_input(self, x):
        x = self.normalize_input(x)
        x = x - x.flip(2)
        return x
    def name_appendix(self):
        return "baseline-mirror" 


class Axial2DCNN(Model):
    def process_input(self, x):
        assert x.shape[-1] == 1, "Axial2DCNN.process_input: expected only 1 axial slice"
        return self.normalize_input(x).squeeze(-1)
    def set_slice_info(self, slice_range, height):
        self.slice_range = slice_range
        self.height      = height
    def name_appendix(self):
        return f"Axial{self.slice_range}{self.height}-2DCNN"
        
        
class UnSup3DCNN(Model):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder)
        assert self.encoder.global_pool is not None
        self.mlp = Encoder("projection_head",
                          torch.nn.Sequential(
                                torch.nn.Linear(encoder.out_channels, 512),
                                torch.nn.ReLU(inplace = True),
                                torch.nn.Linear(512, 128)),
                          128, global_pool = None, dim = 1)      
    def process_input(self, x):
        return self.normalize_input(x)
    def name_appendix(self):
        return "UnSup"
