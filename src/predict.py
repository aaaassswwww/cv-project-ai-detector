from modulefinder import test
from PIL import Image
from sklearn.utils import shuffle
import torch
import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from networks.ssp import ssp
from dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from patch import patch_img

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
parser.add_argument('--split', default='train', type=str, help="dataset split in ['val', 'test']")
parser.add_argument('--model_name', default='ssp', type=str)
parser.add_argument('--model_path', default="./checkpoints", help='Pretrained Model Path')
parser.add_argument('--output_file', default="./result.csv", help='PKL for evaluation')
parser.add_argument('--image_size', default=256, type=int, help='image size')
parser.add_argument('--batch_size', default=50, type=int)
args = parser.parse_args()

def load_model(model_path, device):
    model = ssp()
    model_path = os.path.join(args.model_path, args.model_name)
    model_path = os.path.join(model_path, 'ai-detector_best.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    return model


def predict(model, image, device):
    model.eval()
    class_names = [0,1]
    image = image.to(device)
    output = model(image).ravel()
    
    print(output)
    probability = torch.sigmoid(output)
    print(probability)
    # real_prob = 1 - probability
    # ai_prob = probability
    # confidence = max(real_prob, ai_prob)
    prediction = 1 if probability.item() > 0.5 else 0
    return prediction

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)

    patch_fun = transforms.Lambda(
        lambda img: patch_img(img, 32, 256)
        )
    test_transform = transforms.Compose([
        patch_fun,
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet统计量
        ])
    root_images = os.path.join(args.dataset_root, args.split)
    class_dir = os.path.join(root_images, '1_fake')
    image_path = os.path.join(class_dir, 'fake-0886.png')
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(0)
    print(image.shape)
    prediction = predict(model, image, device)
    print(prediction)
