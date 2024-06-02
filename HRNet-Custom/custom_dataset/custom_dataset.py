import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root, list_path, num_classes, multi_scale=True, flip=True, ignore_label=255,
                 base_size=520, crop_size=(480, 480), downsample_rate=1, scale_factor=16):
        super(CustomDataset, self).__init__()
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.ignore_label = ignore_label
        self.base_size = base_size
        self.crop_size = crop_size
        self.downsample_rate = downsample_rate
        self.scale_factor = scale_factor

        with open(os.path.join(root, list_path), 'r') as f:
            self.ann = json.load(f)

        self.images = self.ann['images']
        self.annotations = self.ann['annotations']
        self.categories = self.ann['categories']

        self.image_transform = transforms.Compose([
            transforms.Resize(self.crop_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_info = self.images[index]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        ann_ids = [ann['id'] for ann in self.annotations if ann['image_id'] == img_info['id']]
        anns = [ann for ann in self.annotations if ann['id'] in ann_ids]

        mask = np.zeros((self.crop_size[0], self.crop_size[1]), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.ann_to_mask(ann))

        image = self.image_transform(image)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def ann_to_mask(self, ann):
        # Convert annotation to binary mask
        mask = Image.new('L', (ann['width'], ann['height']), 0)
        for seg in ann['segmentation']:
            if len(seg) > 0:
                poly = np.array(seg).reshape((len(seg) // 2, 2))
                ImageDraw.Draw(mask).polygon([tuple(p) for p in poly], outline=1, fill=1)
        mask = np.array(mask, dtype=np.uint8)
        return mask
