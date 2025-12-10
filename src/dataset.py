import os
import torch

from PIL import Image

import torch.utils.data as data


class Dataset(data.Dataset):
    image_size = 448

    def __init__(self, args, split, transforms):
        print('DATASET INITIALIZATION')
        self.args = args
        self.split = split  # 保存传入的 split 参数
        root = args.dataset_root
        self.root_images = os.path.join(root, split)
        if split == "train":
            self.train = True
        else:
            self.train = False

        self.transforms = transforms
        self.image_paths = []
        self.labels = []


        self.class_to_idx = {'0_real': 0, '1_fake': 1}
        self.load_data()


    def load_data(self):
        allowed = {'.jpg','.jpeg','.png','.bmp'}
        if self.split == 'test':  # 使用 self.split 而非 self.args.split
            class_dir = self.root_images
            for filename in os.listdir(class_dir):
                if os.path.splitext(filename)[1].lower() not in allowed:
                    continue
                img_path = os.path.join(class_dir, filename)
                self.image_paths.append(img_path)
                self.labels.append(filename)
        else:
            for class_name, label in self.class_to_idx.items():
                class_dir = os.path.join(self.root_images, class_name)

                for filename in os.listdir(class_dir):
                    if os.path.splitext(filename)[1].lower() not in allowed:
                        continue
                    img_path = os.path.join(class_dir, filename)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"错误: 无法加载图像 {img_path}。错误信息: {e}")
            # 返回一个空白图像作为占位符（或抛出异常）
            image = Image.new('RGB', (224, 224), color='black')

        if self.transforms:
            image = self.transforms(image)

        return image, label

    def get_batch(self, batch_indices):
        """
        获取一个批次的图像和标签
        
        Args:
            batch_indices: 图像索引列表
        
        Returns:
            images: 堆叠后的图像张量 (batch_size, C, H, W)
            labels: 对应的标签列表
        """
        images = []
        labels = []
        
        for idx in batch_indices:
            img, label = self.__getitem__(idx)
            images.append(img)
            labels.append(label)
        
        # 将图像堆叠成批次
        if images:
            images = torch.stack(images)
        
        return images, labels
