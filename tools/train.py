# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

from torch.backends import cudnn
from torch import nn
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from utils.logger import setup_logger

from config_hr import config as cfg_hr
from config_hr import update_config


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def train(cfg, cfg_hr):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    # prepare model
    model = build_model(cfg, cfg_hr, num_classes)
    model = nn.DataParallel(model)
    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        optimizer = make_optimizer(cfg, model)
        print(cfg.SOLVER.MARGIN)
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        loss_func = make_loss(cfg, num_classes)     # modified by gu

        arguments = {}

        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query
        )
    elif cfg.MODEL.IF_WITH_CENTER == 'yes':
        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        loss_func, center_criterion = make_loss_with_center(cfg, num_classes)  # modified by gu
        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        arguments = {}

        do_train_with_center(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            num_query
        )
    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(cfg.MODEL.IF_WITH_CENTER))
#    optimizer = make_optimizer(cfg, model)
#    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
#                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
#
#    loss_func = make_loss_with_center(cfg, num_classes)
#
#    arguments = {}
#
#    do_train(
#        cfg,
#        model,
#        train_loader,
#        val_loader,
#        optimizer,
#        scheduler,
#        loss_func,
#        num_query
#    )


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
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
        os.makedirs(output_dir)

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
    train(cfg, cfg_hr)


if __name__ == '__main__':
    main()
