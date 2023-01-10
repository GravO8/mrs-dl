import torch, sys
sys.path.append("..")
from models.encoder import Encoder
from models.models import get_timm_model
from mil_after_max import mil_after_max
from utils.main import main
    

if __name__ == "__main__":
    to_test = [ mil_after_max(get_timm_model("resnet18", 
                                            global_pool = "gap",
                                            pretrained = True,
                                            frozen = False)),
                mil_after_max(get_timm_model("resnet18", 
                                            global_pool = "gap",
                                            pretrained = True,
                                            frozen = True)) ]
    main(to_test)
