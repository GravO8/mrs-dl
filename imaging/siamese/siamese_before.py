import torch, sys
sys.path.append("..")
from models.deep_sym_net import deep_sym_net, deep_sym_merged_encoder
from models.resnet_3d import resnet_3d
from models.siamese import SiameseNetBefore, SiameseEncoder, SiameseL1NormMerger
from models.models import custom_3D_cnn_v1
from utils.main import main

def siamese_before(encoder):
    assert encoder.global_pool is None
    merger          = SiameseL1NormMerger()
    merged_encoder  = deep_sym_merged_encoder(encoder.out_channels, global_pool = "gap")
    siamese_encoder = SiameseEncoder(encoder, merger, merged_encoder)
    return SiameseNetBefore(siamese_encoder)
    
if __name__ == "__main__":
    to_test = [ siamese_before(custom_3D_cnn_v1(global_pool = None)),
                deep_sym_net(),
                siamese_before(resnet_3d(18, global_pool = None)),
                siamese_before(resnet_3d(34, global_pool = None)),
                siamese_before(resnet_3d(50, global_pool = None))]
    main(to_test, device = 1)
