import torch, sys
sys.path.append("..")
from models.deep_sym_net import deep_sym_encoder
from models.resnet_3d import resnet_3d
from models.models import custom_3D_cnn_v1
from models.model import BaselineMirror
from utils.main import main

if __name__ == "__main__":
    # to_test     = [ BaselineMirror(custom_3D_cnn_v1(global_pool = "gap")),
    #                 BaselineMirror(resnet_3d(18, global_pool = "gap")),
    #                 BaselineMirror(resnet_3d(34, global_pool = "gap")),
    #                 BaselineMirror(deep_sym_encoder(1, global_pool = "gap"))]
    to_test = [ BaselineMirror(resnet_3d(50, global_pool = "gap")) ]
    main(to_test)
