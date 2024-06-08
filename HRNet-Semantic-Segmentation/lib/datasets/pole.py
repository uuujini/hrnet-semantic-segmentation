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
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size

        # JSON 파일에서 이미지 리스트를 로드
        self.img_list, self.label_list = self.load_image_and_label_list()

        print(f"Loaded image list: {self.img_list}")  # 디버깅 출력을 추가하여 로드된 이미지 목록 확인

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        if len(self.files) == 0:
            raise ValueError(f"No images found in the dataset path: {os.path.join(root, list_path)}")

        self.class_weights = torch.FloatTensor([1.0] * num_classes)  # class_weights 초기화

    def load_image_and_label_list(self):
        image_list = []
        label_list = []

        with open(os.path.join(self.root, 'annotations', 'instances_default.json'), 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        for img in coco_data['images']:
            # 이미지 파일 경로를 전체 경로로 변환하여 확인
            image_path = os.path.join(self.root, 'images', img['file_name'])
            if os.path.exists(image_path):
                image_list.append(img['file_name'])
                label_list.append(img['id'])

        return image_list, label_list

    def read_files(self):
        files = []
        label_path = os.path.join(self.root, "annotations", "instances_default.json")
        with open(label_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for img, label_id in zip(self.img_list, self.label_list):
                img_file = os.path.join(self.root, 'images', img)
                annotations = [ann for ann in data['annotations'] if ann['image_id'] == label_id]
                files.append({
                    "img": img_file,
                    "label": annotations,
                    "name": os.path.splitext(img)[0]
                })
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = item['img']

        # 경로와 파일 존재 여부 출력
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        size = image.shape

        label = np.zeros(size[:2], dtype=np.uint8)
        for ann in item['label']:
            if 'segmentation' in ann:
                segm = ann['segmentation']
                if isinstance(segm, list):
                    for sub_segm in segm:
                        if isinstance(sub_segm, list):
                            polygon = np.array(sub_segm).reshape(-1, 2)
                            mask = np.zeros(size[:2], dtype=np.uint8)
                            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
                            label = np.maximum(label, mask)

        if 'val' in self.list_path or 'test' in self.list_path:
            image, label = self.resize_short_length(image, label, short_length=self.base_size, fit_stride=8)
            image, label = self.rand_crop(image, label)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), label.copy(), np.array(size), name

        image, label = self.resize_short_length(image, label, short_length=self.base_size)
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def reduce_zero_label(self, labelmap):
        labelmap = np.array(labelmap)
        labelmap[labelmap == 0] = self.ignore_label
        return labelmap
