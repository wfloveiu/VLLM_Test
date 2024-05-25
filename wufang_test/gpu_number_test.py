import torch
import os
# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available. Here are the GPU details:")
    
    # 获取可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    # 打印每个 GPU 的详细信息
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")
