import torch, sys
sys.path.append("..")
from models.deep_sym_net import deep_sym_encoder, deep_sym_merged_encoder
from models.resnet_3d import resnet_3d
from models.siamese import SiameseNetBefore, SiameseTangleMerger, SiameseEncoder
from models.models import custom_3D_cnn_v1, custom_merged_encoder
from utils.main import main

def siamese_tangled(encoder):
    assert encoder.global_pool is None
    merger          = SiameseTangleMerger(in_channels = encoder.out_channels, dim = 3)
    merged_encoder  = deep_sym_merged_encoder(encoder.out_channels, global_pool = "gmp")
    siamese_encoder = SiameseEncoder(encoder, merger, merged_encoder)
    return SiameseNetBefore(siamese_encoder)
    
if __name__ == "__main__":
    # to_test = [ siamese_tangled(custom_3D_cnn_v1(global_pool = None)),
    #             siamese_tangled(deep_sym_encoder(1, global_pool = None)),
    #             siamese_tangled(resnet_3d(18, global_pool = None)),
    #             siamese_tangled(resnet_3d(34, global_pool = None)),
    #             siamese_tangled(resnet_3d(50, global_pool = None))]
    to_test = [ siamese_tangled(resnet_3d(50, global_pool = None)) ]
    main(to_test, device = 0)
