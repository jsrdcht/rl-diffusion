from PIL import Image
import io
import numpy as np
import torch
import os
from accelerate import Accelerator


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def clip_score(accelerator=None):
    """计算图像和文本的CLIP分数作为reward
    使用Hugging Face的CLIP模型计算图像和文本的相似度
    
    Args:
        accelerator: Accelerator实例，如果为None则创建新实例（不推荐）
    """
    from transformers import CLIPProcessor, CLIPModel
    import torch
    
    # 如果没有提供accelerator，创建新实例（不推荐）
    if accelerator is None:
        accelerator = Accelerator()
        print("警告：clip_score函数在没有提供accelerator的情况下创建了新实例，这可能会导致设备分配问题")
    
    # 设置本地模型路径
    local_model_path = "/workspace/ExperimentRecord-main/models/clip-vit-large-patch14"
    
    # 优先从本地加载CLIP模型
    if os.path.exists(local_model_path):
        print("从本地加载CLIP模型...")
        model = CLIPModel.from_pretrained(local_model_path)
        processor = CLIPProcessor.from_pretrained(local_model_path)
    else:
        print("从HuggingFace下载CLIP模型...")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # 保存模型到本地
        print("保存CLIP模型到本地...")
        os.makedirs(local_model_path, exist_ok=True)
        model.save_pretrained(local_model_path)
        processor.save_pretrained(local_model_path)
    
    # 使用accelerator准备模型
    model = accelerator.prepare(model)
    
    def _fn(images, prompts, metadata):
        # 确保prompts是字符串列表
        if not all(isinstance(p, str) for p in prompts):
            prompts = [str(p) for p in prompts]
        
        # 处理图像输入
        if isinstance(images, torch.Tensor):
            # 如果图像是tensor，需要转换格式
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            # 修正transpose操作
            images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image.cpu().numpy()) for image in images]
        else:
            # 如果已经是PIL图像列表，直接使用
            images = [Image.fromarray(image) if isinstance(image, np.ndarray) else image for image in images]
        
        # 使用CLIP处理器处理图像和文本
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 使用accelerator准备输入
        inputs = accelerator.prepare(inputs)
        
        # 获取图像和文本特征
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # 归一化特征向量
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度
            similarity = (image_features @ text_features.T).diagonal()
        
        # 返回相似度分数和空元数据
        return similarity.cpu().numpy(), {}
    
    return _fn


def load_pretrained_resnet18(weights_path):
    """加载预训练的ResNet18模型
    
    Args:
        weights_path: 模型权重文件路径
        
    Returns:
        加载好权重的ResNet18模型
    """
    import torchvision.models as models
    import torch.nn as nn
    
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


def feature_similarity_reward(weights_path=None):
    """使用预训练的ResNet18模型提取图像特征并计算每个图像与其他图像的平均余弦相似度作为reward
    
    Args:
        weights_path: 预训练模型权重路径，如果为None则使用默认路径
        
    Returns:
        reward函数，接受images, prompts, metadata作为输入，返回每个图像的reward值
    """
    if weights_path is None:
        weights_path = "/workspace/SSL-Backdoor/results/backog/imagenet100/trigger_14_targeted_0/mocov2_300epoch/checkpoint_0269.pth.tar"
    
    # 加载预训练模型
    model = load_pretrained_resnet18(weights_path)
    model.eval()  # 设置为评估模式
    
    def _fn(images, prompts, metadata):
        del prompts, metadata  # 不使用这些参数
        
        # 确保输入是tensor
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)
            
        # 确保图像在正确的设备上
        device = next(model.parameters()).device
        images = images.to(device)
        images = images.float()
        
        # 提取特征
        with torch.no_grad():
            features = model(images)
            
        # L2归一化
        features = features / features.norm(dim=-1, keepdim=True)
        
        # 计算所有图像对之间的余弦相似度
        similarity_matrix = torch.matmul(features, features.T)
        
        # 计算每个图像与其他图像的平均相似度（排除自身）
        mask = torch.ones_like(similarity_matrix) - torch.eye(similarity_matrix.shape[0], device=device)
        # 对每个图像计算与其他图像的平均相似度
        mean_similarities = (similarity_matrix * mask).sum(dim=1) / mask.sum(dim=1)
        
        return mean_similarities.cpu().numpy(), {}
        
    return _fn
