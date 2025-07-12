## 模拟硬件中的神经网络 - 基于Q_value定点量化
import torch
import numpy as np
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU
import torch.nn.functional as F
import os

class HardwareSimModel:
    """
    硬件模拟模型 - 基于Q_value定点量化
    适配Model_for_MD工程的量化方案
    """
    
    def __init__(self, model_type='3modes', q_value=12):
        """
        初始化硬件模拟模型
        
        Args:
            model_type: '3modes', '5modes', '6modes'
            q_value: 定点量化的小数位数
        """
        self.model_type = model_type
        self.q_value = q_value
        self.scale = 2.0 ** q_value
        self.coe_dir = f"Model_for_MD/coe/coe_{self.model_type}"
        
        # 根据模型类型设置输入输出参数
        self._set_model_params()
        
        # 初始化网络结构
        self._init_network()
        
        # 加载量化权重
        self._load_quantized_weights()
        
    def _set_model_params(self):
        """根据模型类型设置参数"""
        if self.model_type == '3modes':
            self.input_shape = (1, 1, 16, 16)
            self.output_features = 5
        elif self.model_type == '5modes':
            self.input_shape = (1, 1, 32, 32)
            self.output_features = 9
        elif self.model_type == '6modes':
            self.input_shape = (1, 1, 32, 32)
            self.output_features = 11
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _init_network(self):
        """初始化网络结构"""
        if self.model_type == '3modes':
            self._init_3modes_network()
        elif self.model_type == '5modes':
            self._init_5modes_network()
        elif self.model_type == '6modes':
            self._init_6modes_network()
    
    def _init_3modes_network(self):
        """初始化3modes网络"""
        # 特征提取层
        self.conv1 = Conv2d_Hardware(name="conv1", in_channels=1, out_channels=32, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu1 = ReLU_Hardware()
        self.conv2 = Conv2d_Hardware(name="conv2", in_channels=32, out_channels=32, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu2 = ReLU_Hardware()
        self.maxpool1 = MaxPool2d_Hardware(kernel_size=2)
        self.relu3 = ReLU_Hardware()
        self.conv3 = Conv2d_Hardware(name="conv3", in_channels=32, out_channels=32, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu4 = ReLU_Hardware()
        self.maxpool2 = MaxPool2d_Hardware(kernel_size=2)
        
        # 全连接层
        self.flatten = Flatten_Hardware()
        self.linear1 = Linear_Hardware(name="linear1", in_features=32*4*4, 
                                      out_features=512, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu5 = ReLU_Hardware()
        self.linear2 = Linear_Hardware(name="linear2", in_features=512, 
                                      out_features=5, q_value=self.q_value, coe_dir=self.coe_dir)
    
    def _init_5modes_network(self):
        """初始化5modes网络"""
        # 特征提取层
        self.conv1 = Conv2d_Hardware(name="conv1", in_channels=1, out_channels=32, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu1 = ReLU_Hardware()
        self.conv2 = Conv2d_Hardware(name="conv2", in_channels=32, out_channels=32, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu2 = ReLU_Hardware()
        self.maxpool1 = MaxPool2d_Hardware(kernel_size=2)
        self.relu3 = ReLU_Hardware()
        self.conv3 = Conv2d_Hardware(name="conv3", in_channels=32, out_channels=64, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu4 = ReLU_Hardware()
        self.conv4 = Conv2d_Hardware(name="conv4", in_channels=64, out_channels=64, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu5 = ReLU_Hardware()
        self.maxpool2 = MaxPool2d_Hardware(kernel_size=2)
        self.relu6 = ReLU_Hardware()
        self.conv5 = Conv2d_Hardware(name="conv5", in_channels=64, out_channels=128, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu7 = ReLU_Hardware()
        self.conv6 = Conv2d_Hardware(name="conv6", in_channels=128, out_channels=128, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu8 = ReLU_Hardware()
        self.maxpool3 = MaxPool2d_Hardware(kernel_size=2)
        
        # 全连接层
        self.flatten = Flatten_Hardware()
        self.linear1 = Linear_Hardware(name="linear1", in_features=128*4*4, 
                                      out_features=2048, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu9 = ReLU_Hardware()
        self.linear2 = Linear_Hardware(name="linear2", in_features=2048, 
                                      out_features=512, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu10 = ReLU_Hardware()
        self.linear3 = Linear_Hardware(name="linear3", in_features=512, 
                                      out_features=9, q_value=self.q_value, coe_dir=self.coe_dir)
    
    def _init_6modes_network(self):
        """初始化6modes网络"""
        # 特征提取层
        self.conv1 = Conv2d_Hardware(name="conv1", in_channels=1, out_channels=32, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu1 = ReLU_Hardware()
        self.conv2 = Conv2d_Hardware(name="conv2", in_channels=32, out_channels=32, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu2 = ReLU_Hardware()
        self.maxpool1 = MaxPool2d_Hardware(kernel_size=2)
        self.relu3 = ReLU_Hardware()
        self.conv3 = Conv2d_Hardware(name="conv3", in_channels=32, out_channels=64, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu4 = ReLU_Hardware()
        self.conv4 = Conv2d_Hardware(name="conv4", in_channels=64, out_channels=64, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu5 = ReLU_Hardware()
        self.maxpool2 = MaxPool2d_Hardware(kernel_size=2)
        self.relu6 = ReLU_Hardware()
        self.conv5 = Conv2d_Hardware(name="conv5", in_channels=64, out_channels=128, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu7 = ReLU_Hardware()
        self.conv6 = Conv2d_Hardware(name="conv6", in_channels=128, out_channels=128, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu8 = ReLU_Hardware()
        self.maxpool3 = MaxPool2d_Hardware(kernel_size=2)
        self.relu9 = ReLU_Hardware()
        self.conv7 = Conv2d_Hardware(name="conv7", in_channels=128, out_channels=256, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu10 = ReLU_Hardware()
        self.conv8 = Conv2d_Hardware(name="conv8", in_channels=256, out_channels=256, 
                                     kernel_size=3, padding=1, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu11 = ReLU_Hardware()
        self.maxpool4 = MaxPool2d_Hardware(kernel_size=2)
        
        # 全连接层
        self.flatten = Flatten_Hardware()
        self.linear1 = Linear_Hardware(name="linear1", in_features=256*2*2, 
                                      out_features=2048, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu12 = ReLU_Hardware()
        self.linear2 = Linear_Hardware(name="linear2", in_features=2048, 
                                      out_features=512, q_value=self.q_value, coe_dir=self.coe_dir)
        self.relu13 = ReLU_Hardware()
        self.linear3 = Linear_Hardware(name="linear3", in_features=512, 
                                      out_features=11, q_value=self.q_value, coe_dir=self.coe_dir)
    
    def _load_quantized_weights(self):
        """加载量化权重"""
        # 权重文件路径
        weight_path = f"Model_for_MD/model_pth/{self.model_type}/for{self.model_type}_quantized_int_weights.pth"
        
        if os.path.exists(weight_path):
            print(f"加载量化权重: {weight_path}")
            self.quantized_weights = torch.load(weight_path, map_location='cpu')
        else:
            print(f"警告: 未找到量化权重文件 {weight_path}")
            print("请先运行 quant_general.py 生成量化权重")
            self.quantized_weights = {}
    
    def forward(self, x):
        """前向传播 - 模拟硬件计算流程"""
        # 输入量化
        x = self._quantize_input(x)
        
        # 根据模型类型执行不同的前向传播
        if self.model_type == '3modes':
            return self._forward_3modes(x)
        elif self.model_type == '5modes':
            return self._forward_5modes(x)
        elif self.model_type == '6modes':
            return self._forward_6modes(x)
    
    def _quantize_input(self, x):
        """输入量化"""
        return (x * self.scale).round().to(torch.int32)
    
    def _dequantize_output(self, x):
        """输出反量化"""
        return x.to(torch.float32) / self.scale
    
    def _forward_3modes(self, x):
        """3modes模型前向传播"""
        # 特征提取
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.maxpool1.forward(x)
        x = self.relu3.forward(x)
        x = self.conv3.forward(x)
        x = self.relu4.forward(x)
        x = self.maxpool2.forward(x)
        
        # 全连接层
        x = self.flatten.forward(x)
        x = self.linear1.forward(x)
        x = self.relu5.forward(x)
        x = self.linear2.forward(x)
        
        # 输出反量化
        return self._dequantize_output(x)
    
    def _forward_5modes(self, x):
        """5modes模型前向传播"""
        # 特征提取
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.maxpool1.forward(x)
        x = self.relu3.forward(x)
        x = self.conv3.forward(x)
        x = self.relu4.forward(x)
        x = self.conv4.forward(x)
        x = self.relu5.forward(x)
        x = self.maxpool2.forward(x)
        x = self.relu6.forward(x)
        x = self.conv5.forward(x)
        x = self.relu7.forward(x)
        x = self.conv6.forward(x)
        x = self.relu8.forward(x)
        x = self.maxpool3.forward(x)
        
        # 全连接层
        x = self.flatten.forward(x)
        x = self.linear1.forward(x)
        x = self.relu9.forward(x)
        x = self.linear2.forward(x)
        x = self.relu10.forward(x)
        x = self.linear3.forward(x)
        
        # 输出反量化
        return self._dequantize_output(x)
    
    def _forward_6modes(self, x):
        """6modes模型前向传播"""
        # 特征提取
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.maxpool1.forward(x)
        x = self.relu3.forward(x)
        x = self.conv3.forward(x)
        x = self.relu4.forward(x)
        x = self.conv4.forward(x)
        x = self.relu5.forward(x)
        x = self.maxpool2.forward(x)
        x = self.relu6.forward(x)
        x = self.conv5.forward(x)
        x = self.relu7.forward(x)
        x = self.conv6.forward(x)
        x = self.relu8.forward(x)
        x = self.maxpool3.forward(x)
        x = self.relu9.forward(x)
        x = self.conv7.forward(x)
        x = self.relu10.forward(x)
        x = self.conv8.forward(x)
        x = self.relu11.forward(x)
        x = self.maxpool4.forward(x)
        
        # 全连接层
        x = self.flatten.forward(x)
        x = self.linear1.forward(x)
        x = self.relu12.forward(x)
        x = self.linear2.forward(x)
        x = self.relu13.forward(x)
        x = self.linear3.forward(x)
        
        # 输出反量化
        return self._dequantize_output(x)


class Conv2d_Hardware:
    """硬件模拟卷积层"""
    
    def __init__(self, name, in_channels, out_channels, kernel_size, padding, q_value, coe_dir):
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.q_value = q_value
        self.scale = 2.0 ** q_value
        
        # 权重和偏置路径
        self.weight_path = f"{coe_dir}/{name}.weight_int8.coe"
        self.bias_path = f"{coe_dir}/{name}.bias_int32.coe"
        
        # 加载权重和偏置
        self.weight = self._load_weight()
        self.bias = self._load_bias()
    
    def _load_weight(self):
        """从COE文件加载权重"""
        try:
            with open(self.weight_path, 'r') as f:
                lines = f.readlines()
            # 解析COE文件格式
            weight_data = []
            for line in lines:
                line = line.strip()
                # 跳过头部信息
                if not line or line.startswith('MEMORY_INITIALIZATION') or line.startswith(';'):
                    continue
                # 去掉末尾的逗号或分号
                if line.endswith(',') or line.endswith(';'):
                    line = line[:-1]
                if not line:
                    continue
                # 解析十六进制数据
                try:
                    value = int(line, 16)
                    if value > 0x7F:  # int8补码
                        value -= 0x100
                    weight_data.append(value)
                except ValueError:
                    continue
            print(f"COE文件 {self.weight_path} 读取到 {len(weight_data)} 个数")
            weight = torch.tensor(weight_data, dtype=torch.int8)
            weight = weight.reshape(self.out_channels, self.in_channels, 
                                  self.kernel_size, self.kernel_size)
            return weight
        except Exception as e:
            print(f"警告: 无法加载权重文件 {self.weight_path}: {e}")
            print("请检查coe文件是否正确")
            # 返回随机权重作为备用，避免None错误
            return torch.randint(-128, 127, (self.out_channels, self.in_channels, 
                                           self.kernel_size, self.kernel_size), 
                               dtype=torch.int8)
    
    def _load_bias(self):
        """从COE文件加载偏置"""
        try:
            with open(self.bias_path, 'r') as f:
                lines = f.readlines()
            # 解析COE文件格式
            bias_data = []
            for line in lines:
                line = line.strip()
                # 跳过头部信息
                if not line or line.startswith('MEMORY_INITIALIZATION') or line.startswith(';'):
                    continue
                # 去掉末尾的逗号或分号
                if line.endswith(',') or line.endswith(';'):
                    line = line[:-1]
                if not line:
                    continue
                # 解析十六进制数据
                try:
                    value = int(line, 16)
                    if value > 0x7FFFFFFF:  # int32补码
                        value -= 0x100000000
                    bias_data.append(value)
                except ValueError:
                    continue
            print(f"COE文件 {self.bias_path} 读取到 {len(bias_data)} 个数")
            bias = torch.tensor(bias_data, dtype=torch.int32)
            return bias
        except Exception as e:
            print(f"警告: 无法加载偏置文件 {self.bias_path}: {e}")
            print("请检查coe文件是否正确")
            # 返回零偏置作为备用，避免None错误
            return torch.zeros(self.out_channels, dtype=torch.int32)
    
    def forward(self, x):
        """前向传播"""
        # 执行卷积运算
        conv = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                     kernel_size=self.kernel_size, padding=self.padding)
        conv.weight.data = self.weight.to(torch.float32)
        conv.bias.data = self.bias.to(torch.float32)
        
        # 卷积运算
        output = conv(x.to(torch.float32))
        
        # 量化输出
        return (output / self.scale).round().to(torch.int32)


class Linear_Hardware:
    """硬件模拟线性层"""
    
    def __init__(self, name, in_features, out_features, q_value, coe_dir):
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.q_value = q_value
        self.scale = 2.0 ** q_value
        
        # 权重和偏置路径
        self.weight_path = f"{coe_dir}/{name}.weight_int8.coe"
        self.bias_path = f"{coe_dir}/{name}.bias_int32.coe"
        
        # 加载权重和偏置
        self.weight = self._load_weight()
        self.bias = self._load_bias()
    
    def _load_weight(self):
        """从COE文件加载权重"""
        try:
            with open(self.weight_path, 'r') as f:
                lines = f.readlines()
            # 解析COE文件格式
            weight_data = []
            for line in lines:
                line = line.strip()
                # 跳过头部信息
                if not line or line.startswith('MEMORY_INITIALIZATION') or line.startswith(';'):
                    continue
                # 去掉末尾的逗号或分号
                if line.endswith(',') or line.endswith(';'):
                    line = line[:-1]
                if not line:
                    continue
                # 解析十六进制数据
                try:
                    value = int(line, 16)
                    if value > 0x7F:  # int8补码
                        value -= 0x100
                    weight_data.append(value)
                except ValueError:
                    continue
            print(f"COE文件 {self.weight_path} 读取到 {len(weight_data)} 个数")
            weight = torch.tensor(weight_data, dtype=torch.int8)
            weight = weight.reshape(self.out_features, self.in_features)
            return weight
        except Exception as e:
            print(f"警告: 无法加载权重文件 {self.weight_path}: {e}")
            print("请检查coe文件是否正确")
            # 返回随机权重作为备用
            return torch.randint(-128, 127, (self.out_features, self.in_features), 
                               dtype=torch.int8)
    
    def _load_bias(self):
        """从COE文件加载偏置"""
        try:
            with open(self.bias_path, 'r') as f:
                lines = f.readlines()
            # 解析COE文件格式
            bias_data = []
            for line in lines:
                line = line.strip()
                # 跳过头部信息
                if not line or line.startswith('MEMORY_INITIALIZATION') or line.startswith(';'):
                    continue
                # 去掉末尾的逗号或分号
                if line.endswith(',') or line.endswith(';'):
                    line = line[:-1]
                if not line:
                    continue
                # 解析十六进制数据
                try:
                    value = int(line, 16)
                    if value > 0x7FFFFFFF:  # int32补码
                        value -= 0x100000000
                    bias_data.append(value)
                except ValueError:
                    continue
            print(f"COE文件 {self.bias_path} 读取到 {len(bias_data)} 个数")
            bias = torch.tensor(bias_data, dtype=torch.int32)
            return bias
        except Exception as e:
            print(f"警告: 无法加载偏置文件 {self.bias_path}: {e}")
            print("请检查coe文件是否正确")
            # 返回零偏置作为备用
            return torch.zeros(self.out_features, dtype=torch.int32)
    
    def forward(self, x):
        """前向传播"""
        # 执行线性运算
        linear = Linear(in_features=self.in_features, out_features=self.out_features)
        linear.weight.data = self.weight.to(torch.float32)
        linear.bias.data = self.bias.to(torch.float32)
        
        # 线性运算
        output = linear(x.to(torch.float32))
        
        # 量化输出
        return (output / self.scale).round().to(torch.int32)


class ReLU_Hardware:
    """硬件模拟ReLU激活函数"""
    
    def forward(self, x):
        """前向传播 - 量化ReLU"""
        # 量化ReLU: 小于0的值设为0
        return torch.where(x < 0, torch.tensor(0, dtype=x.dtype), x)


class MaxPool2d_Hardware:
    """硬件模拟最大池化层"""
    
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
    
    def forward(self, x):
        """前向传播"""
        # 最大池化运算
        output = F.max_pool2d(x.to(torch.float32), self.kernel_size)
        return output.to(x.dtype)


class Flatten_Hardware:
    """硬件模拟展平层"""
    
    def forward(self, x):
        """前向传播"""
        return x.view(x.size(0), -1)


def compare_models(hardware_model, float_model, test_input):
    """比较硬件模型和浮点模型的输出"""
    with torch.no_grad():
        hardware_output = hardware_model.forward(test_input)
        float_output = float_model(test_input)
        
        # 计算误差
        mse = F.mse_loss(hardware_output, float_output)
        mae = F.l1_loss(hardware_output, float_output)
        
        print(f"硬件模型输出: {hardware_output}")
        print(f"浮点模型输出: {float_output}")
        print(f"MSE误差: {mse.item():.8f}")
        print(f"MAE误差: {mae.item():.8f}")
        
        return mse, mae


# 测试代码
if __name__ == "__main__":
    # 测试硬件模拟模型
    print("=== 硬件模拟模型测试 ===")
    
    # 创建硬件模型
    hardware_model = HardwareSimModel(model_type='3modes', q_value=12)
    
    # 创建测试输入
    test_input = torch.randn(1, 1, 16, 16)
    
    # 前向传播
    output = hardware_model.forward(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出: {output}")
    
    print("\n=== 测试完成 ===")


