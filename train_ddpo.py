import torch
import time
import numpy as np

def occupy_gpu_memory(gpu_id, memory_gb):
    """占用指定GPU的显存"""
    device = torch.device(f'cuda:{gpu_id}')
    # 计算需要分配的张量大小（以字节为单位）
    size = int(memory_gb * 1024 * 1024 * 1024 / 4)  # 4 bytes per float32
    # 创建一个二维张量来更精确地控制显存使用
    rows = int(np.sqrt(size))
    cols = size // rows
    tensor = torch.zeros((rows, cols), dtype=torch.float32, device=device)
    return tensor

def main():
    # 指定要占用的GPU和显存大小
    gpu_memory_map = {
        5: 3,  # GPU 5 占用 5GB
        7: 3   # GPU 7 占用 5GB
    }
    
    # 存储占用的张量
    occupied_tensors = {}
    
    try:
        # 占用指定GPU的显存
        for gpu_id, memory_gb in gpu_memory_map.items():
            print(f"正在占用 GPU {gpu_id} 的 {memory_gb}GB 显存...")
            occupied_tensors[gpu_id] = occupy_gpu_memory(gpu_id, memory_gb)
        
        print("显存占用完成，开始定期计算...")
        
        # 定期进行计算
        while True:
            for gpu_id, tensor in occupied_tensors.items():
                # 在GPU上进行一些计算
                result = torch.sum(torch.sin(tensor))
                print(f"GPU {gpu_id} 计算结果: {result.item():.4f}")
            
            time.sleep(1)  # 等待5秒
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 清理显存
        for tensor in occupied_tensors.values():
            del tensor
        torch.cuda.empty_cache()
        print("已释放所有显存")

if __name__ == "__main__":
    main() 