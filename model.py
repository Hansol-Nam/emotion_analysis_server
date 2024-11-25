import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import numpy as np
from transformers import AutoModelForImageClassification, AutoFeatureExtractor


# 클래스 레이블 정의
labels = ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]

# 모델 클래스 정의
class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits  # logits만 반환 (추론 결과)

# PyTorch 모델 로드
model_name = "dima806/facial_emotions_image_detection"
pretrained_model = AutoModelForImageClassification.from_pretrained(model_name)
wrapped_model = WrappedModel(pretrained_model)
wrapped_model.eval()

# 이미지 전처리 파이프라인
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])

# 소프트맥스 함수 정의
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # 안정적인 계산을 위해 최대값을 뺌
    return exp_logits / np.sum(exp_logits)

# 이미지 전처리 함수
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))
    return transform(image_resized).unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

# 모델 추론 함수
def predict_emotion(image_tensor):
    with torch.no_grad():
        pytorch_logits = pretrained_model(image_tensor).logits.numpy().flatten()
        print("PyTorch Logits:", pytorch_logits)
    
    # 소프트맥스 적용
    pytorch_probs = softmax(pytorch_logits)    

    # PyTorch 결과 출력
    print("\nPyTorch Model Prediction:")
    for label, prob in zip(labels, pytorch_probs):
        print(f"{label}: {prob:.4f}")

    print(f"Predicted Emotion (PyTorch): {labels[np.argmax(pytorch_probs)]}")

    return {
        "emotion": labels[np.argmax(pytorch_probs)],
        "logits": pytorch_probs.tolist()
    }
