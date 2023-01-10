import timm, torch
from .encoder import Encoder
    
def resnet_3d(version: int = 18, global_pool = "gap"):
    assert version in (18, 34, 50)
    model_name   = f"resnet{version}"
    model_2d     = timm.create_model(model_name, in_chans = 1, num_classes = 0, global_pool = "")
    model_3d     = get_children(model_2d)
    out_channels = (512,2048)[int(version >= 50)]
    return Encoder(model_name, model_3d, out_channels, global_pool = global_pool, dim = 3)

def get_children(model: torch.nn.Module):
    # Adapted from: https://stackoverflow.com/a/65112132
    if list(model.children()) == []:
        return layer_to_3D(model)
    return create_multi_layer(model)

def create_multi_layer(model: torch.nn.Module):
    layers  = [get_children(child) for child in list(model.children())]
    name    = model.__class__.__name__
    if name == "Sequential":
        return torch.nn.Sequential( *layers )
    if name == "ResNet":
        return torch.nn.Sequential( *layers )
    elif (name == "BasicBlock") or (name == "Bottleneck"):
        return ResidualBlock3D(layers)
    elif name == "SelectAdaptivePool2d":
        return torch.nn.Sequential( *layers[::-1] )
    assert False, f"ResNet3D.create_multi_layer: Unknown multiple layer: {name}"
        
def layer_to_3D(layer):
    if isinstance(layer, torch.nn.Conv2d):
        return torch.nn.Conv3d(in_channels      = layer.in_channels,
                                out_channels    = layer.out_channels,
                                kernel_size     = layer.kernel_size[0],
                                stride          = layer.stride[0],
                                padding         = layer.padding[0],
                                bias            = layer.bias)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        return torch.nn.BatchNorm3d(num_features        = layer.num_features,
                                    eps                 = layer.eps,
                                    momentum            = layer.momentum,
                                    affine              = layer.affine,
                                    track_running_stats = layer.track_running_stats)
    elif isinstance(layer, torch.nn.ReLU):
        return layer
    elif isinstance(layer, torch.nn.MaxPool2d):
        return torch.nn.MaxPool3d(kernel_size   = layer.kernel_size,
                                stride          = layer.stride,
                                padding         = layer.padding,
                                dilation        = layer.dilation,
                                ceil_mode       = layer.ceil_mode)
    elif isinstance(layer, torch.nn.Flatten):
        return layer
    elif isinstance(layer, torch.nn.AdaptiveAvgPool2d):
        return torch.nn.AdaptiveAvgPool3d(output_size = layer.output_size)
    elif isinstance(layer, torch.nn.Linear):
        return layer
    elif isinstance(layer, torch.nn.Identity):
        return layer
    elif isinstance(layer, torch.nn.AvgPool2d):
        return torch.nn.AvgPool3d(kernel_size   = 3,
                                stride          = 2,
                                padding         = 1)
    assert False, f"layer_to_3D: Unknown layer {layer}"

    
class ResidualBlock3D(torch.nn.Module):
    def __init__(self, layers):
        super(ResidualBlock3D, self).__init__()
        n_layers = len(layers)
        if (n_layers == 7) or (n_layers == 10): 
            # 7 and 10 layers for Basic and Bottleneck blocks, respectively
            self.downsample = layers[-1] 
            self.activation = layers[-2]
            layers = layers[:-2]
        elif (n_layers == 6) or (n_layers == 9):
            # 6 and 9 layers for Basic and Bottleneck blocks, respectively
            self.downsample = None
            self.activation = layers[-1]
            layers = layers[:-1]
        else:
            assert False, f"ResidualBlock3D.__init__: invalid number of layers {n_layers} for residual block"
        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, x):
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.activation(x)
        return x


if __name__ == "__main__":
    r = resnet_3d(version = "resnet50")
    print(r)
    # sample = torch.rand(2,1,45*2,180//3,91)
    # sample = torch.rand(2,1,45*2,180,91)
    # r(sample)
    # from torchsummary import summary
    # summary(r, (1, 45, 180, 91))
    
