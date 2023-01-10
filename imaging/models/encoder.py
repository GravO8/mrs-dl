import torch


class Encoder(torch.nn.Module):
    def __init__(self, model_name: str, encoder: torch.nn.Module, out_channels: int, 
    global_pool: str = None, dim: int = None):
        torch.nn.Module.__init__(self)
        self.encoder      = encoder
        self.model_name   = model_name
        self.out_channels = out_channels
        self.global_pool  = global_pool
        self.dim          = dim
        if self.global_pool is None:
            self.pooling  = torch.nn.Identity()
        else:
            assert self.dim in (2,3), "Encoder.__init__: specify dim when setting 'global_pool' (valid dims: {2,3})."
            assert self.global_pool in ("gap", "gmp"), "Encoder.__init__: valid values for 'global_pool': {'gap','gmp'}."
            if self.global_pool == "gap":   # global average pooling
                self.pooling = eval(f"torch.nn.AdaptiveAvgPool{self.dim}d")(1)
            elif self.global_pool == "gmp": # global max pooling
                self.pooling = eval(f"torch.nn.AdaptiveMaxPool{self.dim}d")(1)
    def get_name(self):
        return f"{self.model_name}_{'features' if self.global_pool is None else self.global_pool}_{self.dim}D"
    def forward(self, x):
        x = self.encoder(x)
        x = self.pooling(x).squeeze()
        assert self.out_channels == x.shape[1], f"Encoder.forward: expected {self.out_channels} out features, got {x.shape[1]}."
        return x
