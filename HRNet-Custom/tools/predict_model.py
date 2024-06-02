import torch
from torchvision import transforms
from PIL import Image
import json
import os
import sys

# 하드코딩된 경로들
#MODEL_PATH = r'C:\Users\yujin\HRNet\HRNet-Custom\output\config.yaml\train\best.pth'
MODEL_PATH = r'C:\Users\yujin\HRNet\HRNet-Custom\tools\output\config.yaml\train\tb_log\events.out.tfevents.1717243298.BOOK-OI3V4LFE4P'
IMAGE_ROOT = r'C:\Users\yujin\HRNet\HRNet-Custom\data\pole\images'
CATEGORIES_PATH = r'C:\Users\yujin\HRNet\HRNet-Custom\data\pole\annotations\train_annotations.json'

# train.py 파일이 있는 디렉토리를 PYTHONPATH에 추가
sys.path.append(r'C:\Users\yujin\HRNet\HRNet-Custom\tools')

# train.py 파일에서 SimpleModel 불러오기
from train import SimpleModel

# 카테고리 로드
with open(CATEGORIES_PATH, 'r') as f:
    coco = json.load(f)
categories = {cat['id']: cat['name'] for cat in coco['categories']}

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 모델 학습 시의 이미지 크기와 맞춰야 합니다
    transforms.ToTensor(),
])

def load_model(model_path, num_classes):
    model = SimpleModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, image_path, transform, categories):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 배치 차원을 추가합니다
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    return categories[predicted.item()]

# 모델 로드
num_classes = 5  # 실제 클래스 수에 맞게 수정합니다
model = load_model(MODEL_PATH, num_classes)

# 예측할 이미지 경로
image_path = os.path.join(IMAGE_ROOT, 'DSC_0073.JPG')  # 예시 이미지 파일명

# 예측
predicted_class = predict(model, image_path, transform, categories)
print(f'The predicted class for the image is: {predicted_class}')
