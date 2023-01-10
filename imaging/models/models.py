import torch, timm
from .encoder import Encoder

def final_mlp(in_features, bias = True):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features,1, bias = bias), 
        torch.nn.Sigmoid())
    
def conv_3d(in_channels, out_channels, kernel_size, stride = 1, padding = None, 
    bias = True):
    return torch.nn.Sequential( 
        torch.nn.Conv3d(in_channels     = in_channels,
                        out_channels    = out_channels,
                        kernel_size     = kernel_size,
                        stride          = stride,
                        padding         = (kernel_size-1)//2 if padding is None else padding,
                        bias            = bias),
        torch.nn.BatchNorm3d(num_features = out_channels),
        torch.nn.ReLU(inplace = True))
                                
def conv_2d(in_channels, out_channels, kernel_size, stride = 1, padding = None, 
    bias = True):
    return torch.nn.Sequential( 
        torch.nn.Conv2d(in_channels     = in_channels,
                        out_channels    = out_channels,
                        kernel_size     = kernel_size,
                        stride          = stride,
                        padding         = (kernel_size-1)//2 if padding is None else padding,
                        bias            = bias),
        torch.nn.BatchNorm2d(num_features = out_channels),
        torch.nn.ReLU(inplace = True))
        
def custom_3D_cnn_v1(global_pool: str):
    return Encoder("custom_cnn_v1", 
                   torch.nn.Sequential(conv_3d(1,8,5), conv_3d(8,16,3,2,1), 
                                       conv_3d(16,32,3), conv_3d(32,64,3,2,1)),
                   out_channels = 64, 
                   global_pool = global_pool, 
                   dim = 3)
                   
def custom_2D_cnn_v1(global_pool: str):
    return Encoder("custom_cnn_v1", 
                   torch.nn.Sequential(conv_2d(1,8,5), conv_2d(8,16,3,2,1), 
                                       conv_2d(16,32,3), conv_2d(32,64,3,2,1)),
                   out_channels = 64, 
                   global_pool = global_pool, 
                   dim = 2)
                   
def custom_merged_encoder(in_channels: int, global_pool: str):
    return Encoder("custom_merged_encoder", 
                   torch.nn.Sequential(conv_3d(in_channels,64,3), conv_3d(64,128,1)),
                   out_channels = 128,
                   global_pool = global_pool, 
                   dim = 3)
                   
def get_timm_model(model_name: str, global_pool: str = None, 
    pretrained: bool = False, frozen: bool = False, **kwargs):
    appendix = "_pretrained" if pretrained else ""
    supported_models = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "efficientnet_b0": 1280, "efficientnet_b1": 1280}
    assert model_name in supported_models, f"get_timm_model: supported models are {[r for r in supported_models]}"
    model = timm.create_model(model_name, global_pool = "", num_classes = 0, in_chans = 1, pretrained = pretrained, **kwargs)
    if frozen:
        appendix = "_frozen"
        assert pretrained, "get_timm_model: 'frozen' = True only available for 'pretrained' = True"
        for param in model.parameters():
            param.requires_grad = False
    return Encoder( model_name + appendix, 
                    model, 
                    out_channels = supported_models[model_name], 
                    global_pool  = global_pool,
                    dim          = 2)
                    
                    
if __name__ == '__main__':
    # r = get_timm_model("resnet18", global_pool = "gap")
    r = timm.create_model("resnet50")
    print(r)
