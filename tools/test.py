# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn
from torch import nn
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger

from config_hr import config as cfg_hr
from config_hr import update_config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="configs/softmax_triplet_with_center.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument(
             "--cfg", default="configs/cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml", help="path to config file", type=str
                        )
    args = parser.parse_args()
    update_config(cfg_hr, args)
  
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, cfg_hr, num_classes)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))

    inference(cfg, model, val_loader, num_query)


if __name__ == '__main__':
    main()

