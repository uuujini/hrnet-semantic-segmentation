import argparse
import os
import torch
import models
import datasets
from config import config, update_config
from core.function import validate
from utils.utils import create_logger, FullModel

def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--model-file',
                        help='model state file',
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
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test')

    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

    # load model
    model_state_file = args.model_file
    if os.path.isfile(model_state_file):
        logger.info("=> loading model from {}".format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    else:
        raise ValueError("No model found at {}".format(model_state_file))

    model = FullModel(model)
    model = torch.nn.DataParallel(model, device_ids=list(config.GPUS)).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=config.TEST.NUM_SAMPLES,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    if len(test_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the dataset path and contents.")

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    validate(config, testloader, model)

if __name__ == '__main__':
    main()
