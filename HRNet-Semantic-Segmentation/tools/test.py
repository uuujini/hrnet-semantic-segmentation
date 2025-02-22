import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--image',
                        help='path to the test image',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    start = timeit.default_timer()
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus)
    model.eval()

    # Load the test image
    image_path = args.image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(image, (config.TEST.IMAGE_SIZE[0], config.TEST.IMAGE_SIZE[1]))
    input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
    input_image = torch.from_numpy(input_image).unsqueeze(0).float()  # Add batch dimension

    with torch.no_grad():
        input_image = input_image
        output = model(input_image)
        _, pred = torch.max(output, 1)
        pred = pred.cpu().numpy().squeeze()

    # Save the segmentation result
    save_path = os.path.join(final_output_dir, 'segmentation_result.png')
    cv2.imwrite(save_path, pred)
    logger.info(f'Segmentation result saved to {save_path}')

    end = timeit.default_timer()
    logger.info('Mins: %d' % int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()