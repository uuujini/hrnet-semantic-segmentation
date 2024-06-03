import os
import json
import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

class PoleDataset(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=5,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(PoleDataset, self).__init__(ignore_label, base_size,
                                          crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        image_dir = os.path.join(root, list_path)
        if not os.path.isdir(image_dir):
            raise ValueError(f"Provided path is not a directory: {image_dir}")

        self.img_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        if len(self.files) == 0:
            raise ValueError(f"No images found in the dataset path: {image_dir}")

        self.class_weights = torch.FloatTensor([1.0] * num_classes)  # class_weights 초기화 (cuda() 제거)

    def read_files(self):
        files = []
        label_path = os.path.join(self.root, "annotations", "instances_default.json")
        with open(label_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for img in data['images']:
                img_file = os.path.join(self.root, self.list_path, img['file_name'])
                if img_file in self.img_list:
                    img_id = img['id']
                    annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
                    files.append({
                        "img": img_file,
                        "label": annotations,
                        "name": os.path.splitext(os.path.basename(img_file))[0]
                    })
        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name

        label = np.zeros(size[:2], dtype=np.uint8)
        for ann in item['label']:
            if 'segmentation' in ann:
                segm = ann['segmentation']
                if isinstance(segm, list):
                    # Segmentation data를 처리하는 부분
                    for sub_segm in segm:
                        if isinstance(sub_segm, list):
                            polygon = np.array(sub_segm).reshape(-1, 2)
                            mask = np.zeros(size[:2], dtype=np.uint8)
                            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
                            label = np.maximum(label, mask)

        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name
