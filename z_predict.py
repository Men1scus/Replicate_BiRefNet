import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from models.birefnet import BiRefNet
from config import Config

config = Config()
device = config.device

def array_to_pil_image(image, size=(1024, 1024)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(image).convert('RGB')
    return image

class ImagePreprocessor():
    def __init__(self, resolution=(1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image):
        image = self.transform_image(image)
        return image

def load_model(state_dict_path):
    model = BiRefNet(bb_pretrained=False)
    if os.path.exists(state_dict_path):
        birefnet_dict = torch.load(state_dict_path, map_location="cpu")
        unwanted_prefix = '_orig_mod.'
        for k, v in list(birefnet_dict.items()):
            if k.startswith(unwanted_prefix):
                birefnet_dict[k[len(unwanted_prefix):]] = birefnet_dict.pop(k)
        model.load_state_dict(birefnet_dict)
    model = model.to(device)
    model.eval()
    return model

def predict_single_image(model, image_path, resolution=(1024, 1024)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    
    original_shape = image.shape[:2]
    image_pil = array_to_pil_image(image, resolution)

    image_preprocessor = ImagePreprocessor(resolution=resolution)
    image_proc = image_preprocessor.proc(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        scaled_pred_tensor = model(image_proc)[-1].sigmoid()
    pred = torch.nn.functional.interpolate(
        scaled_pred_tensor, size=original_shape, mode='bilinear', align_corners=True
    ).squeeze().cpu().numpy()

    pred_image = np.repeat(np.expand_dims(pred, axis=-1), 3, axis=-1) * 255
    pred_image = pred_image.astype(np.uint8)

    return pred_image

def save_image(image, save_path):
    cv2.imwrite(save_path, image)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Single image inference with BiRefNet')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output image')
    parser.add_argument('--resolution', type=str, default='1024x1024', help='Resolution in WxH format')

    args = parser.parse_args()

    resolution = tuple(map(int, args.resolution.split('x')))
    model = load_model(args.ckpt)
    pred_image = predict_single_image(model, args.image_path, resolution)
    save_image(pred_image, args.output_path)
