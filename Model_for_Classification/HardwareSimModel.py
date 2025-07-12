import torch
import numpy as np
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU
import torch.nn.functional as F
import os

class HardwareSimModel:
    """
    硬件模拟模型 - 基于Q_value定点量化
    适配Model_for_Classification工程的量化方案
    """
    def __init__(self, num_modes=12, q_value=10):
        self.num_modes = num_modes
        self.q_value = q_value
        self.scale = 2.0 ** q_value
        self.coe_dir = f"Model_for_Classification/coe/coe_{num_modes}modes"
        self._init_network()

    def _init_network(self):
        # features: 3个卷积层+ReLU+池化
        self.conv1 = Conv2d_Hardware(name="conv1", in_channels=1, out_channels=32, kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu1 = ReLU_Hardware()
        self.conv2 = Conv2d_Hardware(name="conv2", in_channels=32, out_channels=32, kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu2 = ReLU_Hardware()
        self.maxpool1 = MaxPool2d_Hardware(kernel_size=2)
        self.relu3 = ReLU_Hardware()
        self.conv3 = Conv2d_Hardware(name="conv3", in_channels=32, out_channels=32, kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu4 = ReLU_Hardware()
        self.maxpool2 = MaxPool2d_Hardware(kernel_size=2)
        # 全连接层
        self.flatten = Flatten_Hardware()
        self.linear1 = Linear_Hardware(name="linear1", in_features=32*4*4, out_features=512, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu5 = ReLU_Hardware()
        self.linear2 = Linear_Hardware(name="linear2", in_features=512, out_features=self.num_modes, q_value=self.q_value, coe_dir=self.coe_dir)

    def forward(self, x):
        x = self._quantize_input(x)
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.maxpool1.forward(x)
        x = self.relu3.forward(x)
        x = self.conv3.forward(x)
        x = self.relu4.forward(x)
        x = self.maxpool2.forward(x)
        x = self.flatten.forward(x)
        x = self.linear1.forward(x)
        x = self.relu5.forward(x)
        x = self.linear2.forward(x)
        return self._dequantize_output(x)

    def _quantize_input(self, x):
        return (x * self.scale).round().to(torch.int32)

    def _dequantize_output(self, x):
        return x.to(torch.float32) / self.scale

class Conv2d_Hardware:
    def __init__(self, name, in_channels, out_channels, kernel_size, padding, q_value, coe_dir):
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.q_value = q_value
        self.scale = 2.0 ** q_value
        self.weight_path = f"{coe_dir}/{name}.weight_int8.coe"
        self.bias_path = f"{coe_dir}/{name}.bias_int32.coe"
        self.weight = self._load_weight()
        self.bias = self._load_bias()
    def _load_weight(self):
        try:
            with open(self.weight_path, 'r') as f:
                lines = f.readlines()
            weight_data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('MEMORY_INITIALIZATION') or line.startswith(';'):
                    continue
                if line.endswith(',') or line.endswith(';'):
                    line = line[:-1]
                if not line:
                    continue
                try:
                    value = int(line, 16)
                    if value > 0x7F:
                        value -= 0x100
                    weight_data.append(value)
                except ValueError:
                    continue
            weight = torch.tensor(weight_data, dtype=torch.int8)
            weight = weight.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            return weight
        except Exception as e:
            print(f"警告: 无法加载权重文件 {self.weight_path}: {e}")
            return torch.randint(-128, 127, (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), dtype=torch.int8)
    def _load_bias(self):
        try:
            with open(self.bias_path, 'r') as f:
                lines = f.readlines()
            bias_data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('MEMORY_INITIALIZATION') or line.startswith(';'):
                    continue
                if line.endswith(',') or line.endswith(';'):
                    line = line[:-1]
                if not line:
                    continue
                try:
                    value = int(line, 16)
                    if value > 0x7FFFFFFF:
                        value -= 0x100000000
                    bias_data.append(value)
                except ValueError:
                    continue
            bias = torch.tensor(bias_data, dtype=torch.int32)
            return bias
        except Exception as e:
            print(f"警告: 无法加载偏置文件 {self.bias_path}: {e}")
            return torch.zeros(self.out_channels, dtype=torch.int32)
    def forward(self, x):
        conv = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding)
        conv.weight.data = self.weight.to(torch.float32)
        conv.bias.data = self.bias.to(torch.float32)
        output = conv(x.to(torch.float32))
        return (output / self.scale).round().to(torch.int32)

class Linear_Hardware:
    def __init__(self, name, in_features, out_features, q_value, coe_dir):
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.q_value = q_value
        self.scale = 2.0 ** q_value
        self.weight_path = f"{coe_dir}/{name}.weight_int8.coe"
        self.bias_path = f"{coe_dir}/{name}.bias_int32.coe"
        self.weight = self._load_weight()
        self.bias = self._load_bias()
    def _load_weight(self):
        try:
            with open(self.weight_path, 'r') as f:
                lines = f.readlines()
            weight_data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('MEMORY_INITIALIZATION') or line.startswith(';'):
                    continue
                if line.endswith(',') or line.endswith(';'):
                    line = line[:-1]
                if not line:
                    continue
                try:
                    value = int(line, 16)
                    if value > 0x7F:
                        value -= 0x100
                    weight_data.append(value)
                except ValueError:
                    continue
            weight = torch.tensor(weight_data, dtype=torch.int8)
            weight = weight.reshape(self.out_features, self.in_features)
            return weight
        except Exception as e:
            print(f"警告: 无法加载权重文件 {self.weight_path}: {e}")
            return torch.randint(-128, 127, (self.out_features, self.in_features), dtype=torch.int8)
    def _load_bias(self):
        try:
            with open(self.bias_path, 'r') as f:
                lines = f.readlines()
            bias_data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('MEMORY_INITIALIZATION') or line.startswith(';'):
                    continue
                if line.endswith(',') or line.endswith(';'):
                    line = line[:-1]
                if not line:
                    continue
                try:
                    value = int(line, 16)
                    if value > 0x7FFFFFFF:
                        value -= 0x100000000
                    bias_data.append(value)
                except ValueError:
                    continue
            bias = torch.tensor(bias_data, dtype=torch.int32)
            return bias
        except Exception as e:
            print(f"警告: 无法加载偏置文件 {self.bias_path}: {e}")
            return torch.zeros(self.out_features, dtype=torch.int32)
    def forward(self, x):
        linear = Linear(in_features=self.in_features, out_features=self.out_features)
        linear.weight.data = self.weight.to(torch.float32)
        linear.bias.data = self.bias.to(torch.float32)
        output = linear(x.to(torch.float32))
        return (output / self.scale).round().to(torch.int32)

class ReLU_Hardware:
    def forward(self, x):
        return torch.where(x < 0, torch.tensor(0, dtype=x.dtype), x)

class MaxPool2d_Hardware:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
    def forward(self, x):
        output = F.max_pool2d(x.to(torch.float32), self.kernel_size)
        return output.to(x.dtype)

class Flatten_Hardware:
    def forward(self, x):
        return x.view(x.size(0), -1)

if __name__ == "__main__":
    print("=== 硬件模拟模型测试 ===")
    num_modes = 12
    q_value = 10
    hardware_model = HardwareSimModel(num_modes=num_modes, q_value=q_value)
    test_input = torch.randn(1, 1, 16, 16)
    output = hardware_model.forward(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出: {output}")
    print("\n=== 测试完成 ===") 