import logging
import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(log_path: str, name: str = "train"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.propagate = False
    return logger


def make_worker_init_fn(base_seed: int):
    def seed_worker(worker_id: int):
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed_worker


def predict(model, image, device):
    """
    Predict image class using the model.
    
    Args:
        model: Neural network model
        image: Input tensor (already preprocessed)
        device: Device to run the model on (cpu or cuda)
    
    Returns:
        prediction: 0 (real) or 1 (AI-generated)
    """
    model.eval()
    image = image.to(device)
    output = model(image).ravel()
    probability = torch.sigmoid(output)
    prediction = 1 if probability.item() > 0.5 else 0
    return prediction
