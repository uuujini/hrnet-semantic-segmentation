import argparse
import os
import logging
import time
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
import h5py

# 하드코딩된 데이터셋 경로
TRAIN_ANNOTATIONS = r'C:\Users\yujin\HRNet\HRNet-Custom\data\pole\annotations\train_annotations.json'
IMAGE_ROOT = r'C:\Users\yujin\HRNet\HRNet-Custom\data\pole\images'
TEST_ANNOTATIONS = r'C:\Users\yujin\HRNet\HRNet-Custom\data\pole\annotations\test_annotations.json'


# 커스텀 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, root, list_path, transform=None):
        self.root = root
        with open(list_path, 'r') as f:
            coco = json.load(f)
        self.annotations = coco['annotations']
        self.images = {img['id']: img for img in coco['images']}
        self.categories = {cat['id']: cat['name'] for cat in coco['categories']}
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_info = self.images[annotation['image_id']]
        img_path = os.path.join(self.root, image_info['file_name'])
        label = annotation['category_id']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# 설정 클래스
class Config:
    def __init__(self):
        self.TRAIN = self.TrainConfig()
        self.TEST = self.TestConfig()
        self.DATASET = self.DatasetConfig()
        self.CUDNN = self.CudnnConfig()

    class TrainConfig:
        def __init__(self):
            self.BATCH_SIZE_PER_GPU = 8
            self.IMAGE_SIZE = [256, 256]
            self.MULTI_SCALE = False
            self.FLIP = False
            self.IGNORE_LABEL = -1
            self.BASE_SIZE = 512
            self.DOWNSAMPLERATE = 1
            self.SCALE_FACTOR = 0.25
            self.SHUFFLE = True
            self.LR = 0.01
            self.MOMENTUM = 0.9
            self.WD = 0.0005
            self.NESTEROV = False
            self.END_EPOCH = 1  # 에포크 수 줄이기
            self.EXTRA_EPOCH = 0

    class TestConfig:
        def __init__(self):
            self.IMAGE_SIZE = [256, 256]
            self.NUM_SAMPLES = None
            self.BASE_SIZE = 512

    class DatasetConfig:
        def __init__(self):
            self.NUM_CLASSES = 100  # 데이터셋의 클래스 수에 맞게 수정
            self.ROOT = IMAGE_ROOT

    class CudnnConfig:
        def __init__(self):
            self.BENCHMARK = True
            self.DETERMINISTIC = False
            self.ENABLED = True


config = Config()


# 간단한 모델 클래스
class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 128 * 128, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# CrossEntropy 클래스 정의
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def forward(self, score, target):
        return self.criterion(score, target)


def train(config, epoch, trainloader, optimizer, model, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 10 == 0:
            print(f'Epoch [{epoch}], Step [{i}/{len(trainloader)}], Loss: {loss.item():.4f}')
    accuracy = 100. * correct / total
    return total_loss / len(trainloader), accuracy


def validate(config, testloader, model, criterion, device):
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    return valid_loss / len(testloader), accuracy


def create_logger(config, cfg_name, phase='train'):
    root_output_dir = Path('./output')
    final_output_dir = root_output_dir / cfg_name / phase
    print(f'Creating {final_output_dir}')
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}.log'.format(cfg_name)
    final_log_file = final_output_dir / log_file
    logging.basicConfig(filename=str(final_log_file),
                        format='%(asctime)-15s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    tb_log_dir = final_output_dir / "tb_log"
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tb_log_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--seed', type=int, default=304)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed > 0:
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(config, "config.yaml", 'train')

    logger.info(args)
    logger.info(vars(config))

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn 관련 설정
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleModel(num_classes=config.DATASET.NUM_CLASSES).to(device)

    # GPU가 없는 경우 batch_size 설정
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * max(1, len(gpus))

    transform = transforms.Compose([
        transforms.Resize((config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(
        root=IMAGE_ROOT,
        list_path=TRAIN_ANNOTATIONS,
        transform=transform)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    test_dataset = CustomDataset(
        root=IMAGE_ROOT,
        list_path=TEST_ANNOTATIONS,
        transform=transform)

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL)

    model = nn.DataParallel(model, device_ids=gpus).to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=config.TRAIN.LR,
                          momentum=config.TRAIN.MOMENTUM,
                          weight_decay=config.TRAIN.WD,
                          nesterov=config.TRAIN.NESTEROV)

    best_mIoU = 0
    last_epoch = 0

    start = time.time()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH

    for epoch in range(last_epoch, end_epoch):
        train_loss, train_acc = train(config, epoch, trainloader, optimizer, model, criterion, device)
        valid_loss, valid_acc = validate(config, testloader, model, criterion, device)

        print(
            f'Epoch [{epoch}/{end_epoch}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.2f}%')

        if True:  # args.local_rank <= 0
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

            msg = f'Epoch [{epoch}/{end_epoch}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.2f}%'
            logger.info(msg)

    if True:  # args.local_rank <= 0
        torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'final_state.pth'))

        # 모델을 h5 파일로 저장
        with h5py.File(os.path.join(final_output_dir, 'final_model.h5'), 'w') as f:
            for key, value in model.module.state_dict().items():
                f.create_dataset(key, data=value.cpu().numpy())

        writer_dict['writer'].close()
        end = time.time()
        logger.info('Training completed in: {:.2f} hours'.format((end - start) / 3600))


if __name__ == '__main__':
    main()
