import torch, sys
sys.path.append("..")
from models.encoder import Encoder
from models.models import get_timm_model, custom_2D_cnn_v1
from models.mil import MILEncoder, MILAfterAxial, AttentionMILPooling
from utils.main import main

def mil_after_attention(encoder):
    assert encoder.global_pool is not None
    att_pooling = AttentionMILPooling(in_channels = encoder.out_channels)
    mil_encoder = MILEncoder(encoder = encoder, mil_pooling = att_pooling)
    return MILAfterAxial(mil_encoder)
    

if __name__ == "__main__":
    to_test = [ mil_after_attention(get_timm_model("resnet50", global_pool = "gap", pretrained = False, frozen = False))]
    main(to_test, device = 1)
