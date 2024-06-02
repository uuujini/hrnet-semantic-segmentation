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
                                          crop_size, downsample_rate, scale_factor, mean, std, )

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

        self.class_weights = torch.FloatTensor([1.0] * num_classes)  # 여기서 class_weights 초기화 (cuda() 제거)

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

    def convert_label(self, label, inverse=False):
        return label  # Add conversion logic if needed

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
                    if isinstance(segm[0], str):
                        # 이 경우는 경로가 포함된 리스트
                        mask_path = os.path.join(self.root, segm[0])
                        if os.path.isfile(mask_path):
                            mask = self.convert_label(np.array(Image.open(mask_path)))
                            label = np.maximum(label, mask)
                        else:
                            raise FileNotFoundError(f"Mask file not found: {mask_path}")
                    elif isinstance(segm[0], list):
                        # 이 경우는 좌표가 포함된 다차원 리스트
                        for sub_segm in segm:
                            if isinstance(sub_segm, list):
                                if len(sub_segm) % 2 != 0:
                                    raise ValueError(f"Invalid segmentation sub-item data: {sub_segm}")
                                polygon = np.array(sub_segm).reshape(-1, 2)
                                mask = self.convert_label(np.zeros(size[:2], dtype=np.uint8))
                                cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
                                label = np.maximum(label, mask)
                            else:
                                raise ValueError(f"Invalid segmentation sub-item data: {sub_segm}")
                    else:
                        raise ValueError(f"Invalid segmentation data: {segm}")
                else:
                    raise ValueError(f"Invalid segmentation data: {segm}")
            else:
                raise KeyError(f"Segmentation key not found in annotation: {ann}")

        image, label = self.gen_sample(image, label,
                                       self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = int(self.crop_size[0] * 1.0)
        stride_w = int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width])
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = int(np.ceil(1.0 * (new_h -
                                          self.crop_size[0]) / stride_h)) + 1
                cols = int(np.ceil(1.0 * (new_w -
                                          self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w])
                count = torch.zeros([1, 1, new_h, new_w])

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(h1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
