import numpy as np
import torch
from torch.utils.data import Dataset

class LPModesDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data['X']  # (num_samples, H, W)
        self.y_amp = data['y_amp']  # (num_samples, num_modes)
        self.y_phase = data['y_phase']  # (num_samples, num_modes-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # 转换为 torch.Tensor，并保证数据shape正确
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        y_amp = torch.tensor(self.y_amp[idx], dtype=torch.float32)
        y_phase = torch.tensor(self.y_phase[idx], dtype=torch.float32)
        # 你可以根据实际任务返回标签（如y_amp、y_phase、或拼接）
        label = torch.cat([y_amp, y_phase])
        return x, label
    
class LPModesClassificationDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data['X']  # (num_samples, H, W)
        self.y = data['y']  # (num_samples,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        y = torch.tensor(self.y[idx], dtype=torch.int64)  # 类别标签
        return x, y