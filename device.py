import torch

def get_default_device():
    """
    在 macOS Apple Silicon 上使用 MPS；否则使用 CPU。
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
