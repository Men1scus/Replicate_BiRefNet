# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from cog import BasePredictor, Input, Path

from models.birefnet import BiRefNet
from config import Config

config = Config()
device = config.device

class ImagePreprocessor():
    def __init__(self, resolution=(1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            # transforms.Resize(resolution),    # 1. keep consistent with the cv2.resize used in training 2. redundant with that in path_to_image()
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image):
        image = self.transform_image(image)
        return image

def array_to_pil_image(image, size=(1024, 1024)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(image).convert('RGB')
    return image

def load_model(state_dict_path):
    model = BiRefNet(bb_pretrained=False)
    if os.path.exists(state_dict_path):
        birefnet_dict = torch.load(state_dict_path, map_location="cpu")
        unwanted_prefix = '_orig_mod.'
        # 移除状态字典中不需要的前缀
        for k, v in list(birefnet_dict.items()):
            if k.startswith(unwanted_prefix):
                birefnet_dict[k[len(unwanted_prefix):]] = birefnet_dict.pop(k)
        model.load_state_dict(birefnet_dict)
    model = model.to(device)
    model.eval()
    return model

def predict(model, image, resolution):

    image = cv2.imread(image)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    
    # 把 OpenCV 读取的 BGR 格式图片 转换为 Pytorch 中的 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 用原图的分辨率
    resolution = f"{image.shape[1]}x{image.shape[0]}" if resolution == '' else resolution
    resolution = [int(int(reso)//32*32) for reso in resolution.strip().split('x')]

    images = [image_rgb]
    image_shapes = [image.shape[:2] for image in images]
    images = [array_to_pil_image(image, resolution) for image in images]

    image_preprocessor = ImagePreprocessor(resolution=resolution)
    images_proc = []
    for image in images:
        images_proc.append(image_preprocessor.proc(image))
    images_proc = torch.cat([image_proc.unsqueeze(0) for image_proc in images_proc])

    with torch.no_grad():
        scaled_preds_tensor = model(images_proc.to(device))[-1].sigmoid()
        
    preds = []
    for image_shape, pred_tensor in zip(image_shapes, scaled_preds_tensor):
        if device == 'cuda':
            pred_tensor = pred_tensor.cpu()
        preds.append(torch.nn.functional.interpolate(pred_tensor.unsqueeze(0), size=image_shape, mode='bilinear', align_corners=True).squeeze().numpy())

    image_preds = []
    for image, pred in zip(images, preds):
        image = image.resize(pred.shape[::-1])
        pred = (pred * 255).astype(np.uint8)  # Scale prediction to 0-255

        # Create RGBA image
        rgba_image = np.dstack((np.array(image), pred))
        rgba_image = Image.fromarray(rgba_image, 'RGBA')
        image_preds.append(rgba_image)

    return image_preds[0]

def save_image(image, save_path):
    image.save(save_path)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = load_model("BiRefNet-general-epoch_244.pth")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        resolution: str = Input(
            description="Resolution in WxH format, e.g., '1024x1024'",
            default=""
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        output_image = predict(self.model, str(image), resolution)
        output_path = "output.png"
        save_image(output_image, output_path)
        return Path(output_path)
