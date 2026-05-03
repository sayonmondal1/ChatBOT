import torch
# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available. Training will use GPU.")
else:
    print("GPU not available. Training will use CPU.")