import torch, sys
sys.path.append("..")
from models.models import custom_3D_cnn_v1
from models.mil import MILEncoder, MILAfterBlock, MaxMILPooling, MeanMILPooling, AttentionMILPooling
from utils.main import main

def mil_after_blocks(pooling):
    encoder     = custom_3D_cnn_v1(global_pool = "gap")
    mil_encoder = MILEncoder(encoder = encoder, mil_pooling = pooling)
    return MILAfterBlock(mil_encoder)
    

if __name__ == "__main__":
    to_test = [ mil_after_blocks(MaxMILPooling()),
                mil_after_blocks(MeanMILPooling()),
                mil_after_blocks(AttentionMILPooling(in_channels = 64))]
    main(to_test)
