
from .baseline import Baseline

def build_model(cfg, cfg_stage, num_classes):
    model = Baseline(cfg_stage, num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH)
    return model
    
