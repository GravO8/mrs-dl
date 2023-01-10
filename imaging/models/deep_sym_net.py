import torch
from .encoder import Encoder
from .siamese import SiameseEncoder, SiameseL1NormMerger, SiameseNet
from .models import conv_3d

class InceptionModule3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels: int = 64):
        torch.nn.Module.__init__(self)
        self.path1 = conv_3d(in_channels, out_channels, kernel_size = 1, padding = 0, bias = False)
        self.path2 = torch.nn.Sequential(
                        conv_3d(in_channels,  out_channels, kernel_size = 1, padding = 0, bias = False),
                        conv_3d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False))
        self.path3 = torch.nn.Sequential(
                        conv_3d(in_channels,  out_channels, kernel_size = 1, padding = 0, bias = False),
                        conv_3d(out_channels, out_channels, kernel_size = 5, padding = 2, bias = False))
        self.path4 = torch.nn.Sequential(
                        torch.nn.MaxPool3d(kernel_size  = 3, 
                                           stride       = 1,
                                           padding      = 1,
                                           ceil_mode    = True),
                        conv_3d(in_channels, out_channels, kernel_size = 1, padding = 0, bias = False))
    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)
        x  = torch.cat([x1, x2, x3, x4], dim = 1)  
        return x


def deep_sym_encoder(in_channels: int, global_pool = None):
    encoder = torch.nn.Sequential(
        InceptionModule3D(in_channels,  4),
        torch.nn.AvgPool3d(kernel_size = 2),
        InceptionModule3D(4*4, 16),
        InceptionModule3D(16*4, 16),
        InceptionModule3D(16*4, 16)
    )
    return Encoder("deep_sym_encoder", encoder, 16*4, global_pool = global_pool, dim = 3)
    

def deep_sym_merged_encoder(in_channels: int, global_pool = "gmp"):
    encoder = torch.nn.Sequential(
        InceptionModule3D(in_channels, 16),
        InceptionModule3D(16*4, 16)
    )
    return Encoder("deep_sym_merged_encoder", encoder, 16*4, global_pool = global_pool, dim = 3)


def deep_sym_net(in_channels = 1):
    encoder         = deep_sym_encoder(in_channels, global_pool = None)
    merger          = SiameseL1NormMerger()
    merged_encoder  = deep_sym_merged_encoder(encoder.out_channels)
    return SiameseNet(SiameseEncoder(encoder, merger, merged_encoder))


if __name__ == "__main__":
    deep_sym_encoder = deep_sym_encoder(1)
    x = torch.randn(32,1,91,109,91)
    print(deep_sym_encoder(x).shape)
        
