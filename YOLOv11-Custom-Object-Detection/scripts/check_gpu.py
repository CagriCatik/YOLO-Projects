import torch
import sys

def check_gpu():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
    else:
        print("GPU NOT FOUND. Training will be slow on CPU.")

if __name__ == "__main__":
    check_gpu()
