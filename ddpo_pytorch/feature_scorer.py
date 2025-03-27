from importlib import resources
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torchvision.models as models

ASSETS_PATH = resources.files("ddpo_pytorch.assets")


def load_pretrained_resnet18(weights_path):
    """加载预训练的ResNet18模型
    
    Args:
        weights_path: 模型权重文件路径
        
    Returns:
        加载好权重的ResNet18模型
    """
    
    if weights_path is None:
        weights_path = ASSETS_PATH.joinpath("na_target0.pth.tar")
    
    # 创建ResNet18模型
    model = models.resnet18(pretrained=False)
    
    # 移除最后的全连接层
    model.fc = nn.Identity()
    
    # 加载预训练权重
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # 处理权重字典
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint['state_dict']
    
    # 过滤和重命名权重键
    def is_valid_model_param_key(key):
        valid_keys = ['encoder_q', 'backbone', 'encoder']
        invalid_keys = ['fc', 'head', 'predictor', 'projection_head', 'encoder_k', 'model_t', 'backbone_momentum']
        
        if any([k in key for k in invalid_keys]):
            return False
        if not any([k in key for k in valid_keys]):
            return False
        return True
    
    def model_param_key_filter(key):
        if 'model.' in key:
            key = key.replace('model.', '')
        if 'module.' in key:
            key = key.replace('module.', '')
        if 'encoder.' in key:
            key = key.replace('encoder.', '')
        if 'encoder_q.' in key:
            key = key.replace('encoder_q.', '')
        if 'backbone.' in key:
            key = key.replace('backbone.', '')
        return key
    
    state_dict = {model_param_key_filter(k): v for k, v in state_dict.items() if is_valid_model_param_key(k)}
    
    # 加载处理后的权重
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"加载预训练模型权重: {msg}")
    
    return model