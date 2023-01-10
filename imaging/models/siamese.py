import torch
from .encoder import Encoder
from .model import Model
from abc import ABC, abstractmethod


class SiameseEncoderMerger:
    '''
    Command design pattern
    fn should be a function that receives two inputs tensors, each of size 
    (batch, channels, features_maps+) and outputs a single tensor of the same 
    size whose content is obtained by merging the two input tensors in some way
    '''
    @abstractmethod
    def get_name(self):
        pass
    @abstractmethod
    def __call__(self, x1, x2):
        pass
        
        
def tangle(x1, x2):
    assert x1.shape == x2.shape
    x1 = torch.cat((x1,x1), dim = 1)
    x1[:,[i for i in range(x1.shape[1]) if i%2 == 0],:,:,:] = x1[:,:x1.shape[1]//2,:,:,:]
    x1[:,[i for i in range(x1.shape[1]) if i%2 == 1],:,:,:] = x2
    return x1
        
class SiameseTangleMerger(SiameseEncoderMerger, torch.nn.Module):
    def __init__(self, in_channels: int, dim: int = 3):
        super().__init__()
        assert dim in (2,3)
        self.group_conv = eval(f"torch.nn.Conv{dim}d")(in_channels  = in_channels*2, 
                                                       out_channels = in_channels, 
                                                       groups       = in_channels,
                                                       kernel_size  = 3, 
                                                       padding      = 1)
    def get_name(self):
        return "tangle"
    def __call__(self, x1, x2):
        x = tangle(x1, x2)
        x = self.group_conv(x)
        return x
        
class SiameseL1NormMerger(SiameseEncoderMerger):
    def get_name(self):
        return "L1norm"
    def __call__(self, x1, x2):
        return torch.abs(x1 - x2)
        

class SiameseEncoder(torch.nn.Module):
    def __init__(self, encoder: Encoder, merge_encodings: SiameseEncoderMerger, 
    merged_encoder: Encoder):
        super().__init__()
        self.encoder         = encoder
        self.merge_encodings = merge_encodings
        self.merged_encoder  = merged_encoder
        self.out_channels    = self.merged_encoder.out_channels
        if self.merged_encoder.global_pool is None:
            assert self.encoder.global_pool is not None, "SiameseEncoder.__init__: either the 'encoder' or the 'merged_encoder' must apply global pooling."
        else:
            assert self.encoder.global_pool is None, "SiameseEncoder.__init__: global pooling can't be applied by both the 'encoder' and the 'merged_encoder'."
    def get_name(self):
        return f"{self.encoder.get_name()}_{self.merge_encodings.get_name()}_{self.merged_encoder.get_name()}"
    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x  = self.merge_encodings(x1, x2)
        x  = self.merged_encoder(x) 
        assert self.merged_encoder.out_channels == x.shape[1], f"SiameseEncoder.forward: expected {self.merged_encoder.out_channels} out features, got {x.shape[1]}."
        return x


class SiameseNet(Model):
    def __init__(self, encoder: SiameseEncoder):
        assert isinstance(encoder, SiameseEncoder), "SiameseNet.__init__: 'encoder' must be of class 'SiameseEncoder'"
        super().__init__(encoder)
    def process_input(self, x):
        x           = self.normalize_input(x)
        msp         = x.shape[2]//2             # midsagittal plane
        hemisphere1 = x[:,:,:msp,:,:]           # shape = (B,C,x,y,z)
        hemisphere2 = x[:,:,msp:-1,:,:].flip(2) # B - batch; C - channels
        return (hemisphere1, hemisphere2)
    def forward(self, x):
        x1, x2 = self.process_input(x)
        x      = self.encoder(x1, x2)
        return self.mlp(x)
    def name_appendix(self):
        return "SiameseNet"


class SiameseNetBefore(SiameseNet):
    def __init__(self, encoder: SiameseEncoder):
        super().__init__(encoder)
        assert encoder.encoder.global_pool is None
        assert encoder.merged_encoder.global_pool is not None
    def name_appendix(self):
        return super().name_appendix() + "-before"
        
        
class SiameseNetAfter(SiameseNet):
    def __init__(self, encoder: SiameseEncoder):
        super().__init__(encoder)
        assert encoder.encoder.global_pool is not None
        assert encoder.merged_encoder.global_pool is None
    def name_appendix(self):
        return super().name_appendix() + "-after"
