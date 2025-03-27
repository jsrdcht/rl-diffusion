import torch
import time
import threading
import numpy as np

def cpu_task():
    """CPU密集型计算任务"""
    while True:
        # 大规模矩阵运算，增加计算量
        size = 2000
        for _ in range(5):  # 连续进行多次计算
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            c = np.dot(a, b)
            # 进行更多运算以增加CPU负载
            np.exp(c)
            np.log(np.abs(c) + 1)
            np.sin(c)

def gpu_compute(gpu_id):
    """GPU计算任务"""
    device = f"cuda:{gpu_id}"
    # 预先创建一些大型张量
    a = torch.rand(4096, 4096, device=device)
    b = torch.rand(4096, 4096, device=device)
    
    # 创建一个较大的卷积层
    conv = torch.nn.Conv2d(128, 128, 3, padding=1).to(device)
    
    while True:
        # 连续进行多次大规模矩阵运算
        for _ in range(5):
            c = torch.matmul(a, b)
            d = torch.matmul(c, a)
            # 添加更多运算以增加GPU负载
            torch.sin(d)
            torch.exp(d)
        
        # 进行大规模卷积运算
        input_tensor = torch.randn(16, 128, 256, 256, device=device)
        for _ in range(3):
            output = conv(input_tensor)
            output = torch.relu(output)
            input_tensor = output

def occupy_memory(gpu_id, memory_gb=20):
    """
    在指定GPU上占用指定大小的显存并进行计算
    
    Args:
        gpu_id (int): GPU ID
        memory_gb (int): 要占用的显存大小(GB)
    """
    torch.cuda.set_device(gpu_id)
    print(f"正在GPU {gpu_id}上分配 {memory_gb}GB 显存...")
    
    # 计算需要分配的张量大小
    tensor_size = int(memory_gb * 1024 * 1024 * 1024 / 4)
    
    # 创建并保持张量在GPU上
    tensor = torch.zeros(tensor_size, dtype=torch.float32, device=f"cuda:{gpu_id}")
    
    print(f"GPU {gpu_id} 显存分配完成")
    
    # 启动GPU计算线程
    gpu_thread = threading.Thread(target=gpu_compute, args=(gpu_id,))
    gpu_thread.daemon = True
    gpu_thread.start()
    
    return tensor

if __name__ == "__main__":
    # 启动更多CPU计算线程
    num_cpu_threads = 8  # 增加到8个CPU线程
    cpu_threads = []
    for _ in range(num_cpu_threads):
        thread = threading.Thread(target=cpu_task)
        thread.daemon = True
        thread.start()
        cpu_threads.append(thread)
    
    # 在GPU 5和7上各占用20GB显存并进行计算
    tensors = []
    for gpu_id in [5, 7]:
        tensor = occupy_memory(gpu_id, 20)
        tensors.append(tensor)
    
    print("显存占用和计算任务启动完成，程序将持续运行...")
    print(f"已启动 {num_cpu_threads} 个CPU计算线程")
    print("按Ctrl+C终止程序")
    
    try:
        while True:
            # 每隔5秒打印一次GPU使用情况
            for gpu_id in [5, 7]:
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                print(f"GPU {gpu_id} - 已分配: {memory_allocated:.2f}GB, 已预留: {memory_reserved:.2f}GB")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n收到中断信号，程序退出") 