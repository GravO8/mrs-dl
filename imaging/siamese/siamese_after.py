import torch, sys
sys.path.append("..")
from models.deep_sym_net import deep_sym_encoder
from models.resnet_3d import resnet_3d
from models.encoder import Encoder
from models.siamese import SiameseEncoder, SiameseNetAfter, SiameseL1NormMerger
from models.models import custom_3D_cnn_v1
from utils.main import main

def siamese_after(encoder):
    assert encoder.global_pool is not None
    merger          = SiameseL1NormMerger()
    merged_encoder  = Encoder("MLP_merged_encoder",
                              torch.nn.Sequential(
                                    torch.nn.Linear(encoder.out_channels, 512),
                                    torch.nn.ReLU(inplace = True),
                                    torch.nn.Linear(512, 128),
                                    torch.nn.ReLU(inplace = True)),
                              128, global_pool = None, dim = 1)
    siamese_encoder = SiameseEncoder(encoder, merger, merged_encoder)
    return SiameseNetAfter(siamese_encoder)
    
if __name__ == "__main__":
    # to_test = [ siamese_after(custom_3D_cnn_v1(global_pool = "gap")),
    #             siamese_after(deep_sym_encoder(1, global_pool = "gap")),
    #             siamese_after(resnet_3d(18, global_pool = "gap")),
    #             siamese_after(resnet_3d(34, global_pool = "gap"))]
    to_test = [ siamese_after(custom_3D_cnn_v1(global_pool = "gmp")),
                siamese_after(deep_sym_encoder(1, global_pool = "gmp")),
                siamese_after(resnet_3d(18, global_pool = "gmp")),
                siamese_after(resnet_3d(34, global_pool = "gmp")),
                siamese_after(resnet_3d(50, global_pool = "gmp"))]
    main(to_test, device = 0)
