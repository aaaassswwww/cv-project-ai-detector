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
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
parser.add_argument('--split', default='test', type=str, help="dataset split in ['val', 'test']")
parser.add_argument('--model_name', default='ssp', type=str)
parser.add_argument('--model_path', default="./checkpoints", help='Pretrained Model Path')
parser.add_argument('--output_file', default="./result.csv", help='PKL for evaluation')
parser.add_argument('--image_size', default=256, type=int, help='image size')
parser.add_argument('--batch_size', default=50, type=int)
args = parser.parse_args()


def predict(model, image, device):
    model.eval()
    image = image.to(device)
    output = model(image).ravel()
    
    probability = torch.sigmoid(output)
    # real_prob = 1 - probability
    # ai_prob = probability
    # confidence = max(real_prob, ai_prob)
    prediction = 1 if probability.item() > 0.5 else 0
    return prediction

def save_results(image_ids, predictions, output_file):
    """
    保存预测结果到CSV文件
    """
    # 创建DataFrame
    df = pd.DataFrame({
            'image_id': image_ids,
            'label': predictions
    })
    
    # 保存到CSV文件
    df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")
    
    # 打印统计信息
    print("\n预测结果统计:")
    print(f"总图像数: {len(image_ids)}")
    print(f"预测为真实图像 (0): {predictions.count(0)}")
    print(f"预测为AI生成图像 (1): {predictions.count(1)}")
    
    
    return df

def load_model(model_path, device):
    model = ssp()
    model_path = os.path.join(args.model_path, args.model_name)
    model_path = os.path.join(model_path, 'ai-detector_best.pth')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    return model

def main():
    
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
    image_paths = []
    predictions = []
    filenames = []
    class_dir = root_images
    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)
        image_paths.append(img_path)
        filenames.append(filename)

    image_paths = natsorted(image_paths)
    filenames = natsorted(filenames)


    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        image = test_transform(image)
        image = image.unsqueeze(0)
        prediction = predict(model, image, device)
        predictions.append(prediction)

    save_results(filenames, predictions, args.output_file)

if __name__ == '__main__':
    main()


