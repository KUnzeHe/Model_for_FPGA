import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, ReLU

# 搭建神经网络
class for5modes_net(nn.Module):
    # 初始化方法
    def __init__(self):
        # 继承nn.Module的初始化方法
        super(for5modes_net, self).__init__()

        # features提取层(封装了所有的卷积层、激活函数和池化层)
        # input(32×32 image)
        self.features = Sequential(
            Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            ReLU(),
            Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )

        self.linear = Sequential(
            nn.Flatten(),    # 展平层
            nn.Linear(in_features=128*4*4, out_features=2048),    # 全连接层 FC2048（2048个神经元，输出特征数为 2048）
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=512),    # 全连接层 FC512（512个神经元，输出特征数为 512）
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=9)    # 全连接层 FC9（9个神经元，输出特征数为 9=5*振幅+4*相位）
        )

    def forward(self, x):
        y = self.features(x)  # 特征提取
        y = self.linear(y)    # 全连接层处理
        return y
    
# --- 测试一下代码 ---
if __name__ == '__main__':
    # 创建模型实例，如果代码有错，这里会报错
    model = for5modes_net()
    
    # 打印模型结构，可以看到Sequential的清晰组织
    print(model)
    
    # 创建一个虚拟输入来测试前向传播
    # 假设输入是 32x32 的单通道图片
    dummy_input = torch.randn(1, 1, 32, 32) 
    output = model(dummy_input)
    
    print(f"\n测试输入形状: {dummy_input.shape}")
    print(f"最终输出形状: {output.shape}") # 应该是 [1, 9]
