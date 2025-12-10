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
from utils.patch import patch_img_deterministic
from utils.util import predict, set_seed
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='./dataset', type=str, help='dataset root')
parser.add_argument('--split', default='test', type=str, help="dataset split in ['val', 'test']")
parser.add_argument('--model_name', default='ssp', type=str)
parser.add_argument('--model_path', default="./checkpoints", help='Pretrained Model Path')
parser.add_argument('--output_file', default="./result.csv", help='PKL for evaluation')
parser.add_argument('--image_size', default=256, type=int, help='image size')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')
args = parser.parse_args()


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

def predict_batch(model, images, device):
    """
    批量预测图像类别。
    
    Args:
        model: 模型
        images: 图像 tensor (batch_size, C, H, W)
        device: 设备
    
    Returns:
        predictions: 预测结果列表
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images).ravel()
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).long().cpu().tolist()
    return predictions


def main():
    # 设置随机种子确保可复现性
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = load_model(args.model_path, device)

    # 使用确定性 patch 选择
    patch_fun = transforms.Lambda(
        lambda img: patch_img_deterministic(img, 32, 256)
        )
    test_transform = transforms.Compose([
        patch_fun,
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    root_images = os.path.join(args.dataset_root, args.split)
    image_paths = []
    filenames = []
    
    for filename in os.listdir(root_images):
        img_path = os.path.join(root_images, filename)
        image_paths.append(img_path)
        filenames.append(filename)

    # 自然排序确保顺序一致
    sorted_pairs = natsorted(zip(filenames, image_paths), key=lambda x: x[0])
    filenames, image_paths = zip(*sorted_pairs) if sorted_pairs else ([], [])
    filenames = list(filenames)
    image_paths = list(image_paths)

    # 批量推理
    predictions = []
    batch_size = args.batch_size
    
    print(f"Total images: {len(image_paths)}, Batch size: {batch_size}")
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inference"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            image = Image.open(img_path).convert('RGB')
            image = test_transform(image)
            batch_images.append(image)
        
        batch_tensor = torch.stack(batch_images)
        batch_preds = predict_batch(model, batch_tensor, device)
        predictions.extend(batch_preds)

    save_results(filenames, predictions, args.output_file)

if __name__ == '__main__':
    main()


