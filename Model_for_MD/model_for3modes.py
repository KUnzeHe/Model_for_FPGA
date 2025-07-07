import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU

# 搭建神经网络
class for3modes_net(nn.Module):
    # 初始化方法
    def __init__(self):
        # 继承nn.Module的初始化方法
        super(for3modes_net, self).__init__()

        # features提取层(封装了所有的卷积层、激活函数和池化层)
        self.features = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )

        # 全连接层(线性层)
        self.linear = Sequential(
            nn.Flatten(),    # 展平层
            nn.Linear(in_features=32*4*4, out_features=512),    # 全连接层 FC512（512个神经元，输出特征数为 512）
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=5)    # 全连接层 FC5（5个神经元，输出特征数为 5=3*振幅+2*相位）
        )
    
    def forward(self, x):
       y = self.features(x)  # 特征提取
       y = self.linear(y)    # 全连接层处理 
       return y
    
# --- 测试一下代码 ---
if __name__ == '__main__':
    # 创建模型实例，如果代码有错，这里会报错
    model = for3modes_net()
    
    # 打印模型结构，可以看到Sequential的清晰组织
    print(model)
    
    # 创建一个虚拟输入来测试前向传播
    # 假设输入是 16x16 的单通道图片
    dummy_input = torch.randn(1, 1, 16, 16) 
    output = model(dummy_input)
    
    print(f"\n测试输入形状: {dummy_input.shape}")
    print(f"最终输出形状: {output.shape}") # 应该是 [1, 5]


