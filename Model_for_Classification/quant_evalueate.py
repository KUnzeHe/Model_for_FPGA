import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 导入所有必要的模块和自定义类
# ==============================================================================
# 确保这些模型定义文件和数据集加载文件在同一目录下
from model_forClassification import forClassification_net
from lp_dataset import LPModesClassificationDataset

# ==============================================================================
# 2. 定义核心模型类
# ==============================================================================

class QuantizedNet(nn.Module):
    """
    通用的模拟定点运算模型类。
    它的作用是加载浮点模型的结构，并用Q值量化方案来模拟前向传播。
    """
    def __init__(self, float_model, q_value):
        super().__init__()
        self.features = copy.deepcopy(float_model.features)
        self.linear = copy.deepcopy(float_model.linear)
        self.q_value = q_value
        self.scale = 2.0 ** q_value
        self.float() # 确保模型在浮点模式下

    def forward(self, x):
        # 模拟硬件行为：
        # 1. 输入量化
        x_quant = (x * self.scale).round()
        
        # 2. 逐层计算与再量化
        # 特征提取层
        for layer in self.features:
            x_quant = layer(x_quant)
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # 每次卷积/全连接后，将结果缩放回原始范围
                x_quant = (x_quant / self.scale).round()
        # 全连接层
        for layer in self.linear:
            x_quant = layer(x_quant)
            if isinstance(layer, nn.Linear):
                x_quant = (x_quant / self.scale).round()
        
        # 3. 输出反量化 (关键步骤：将整数结果转换回浮点域，以便与真实标签比较)
        output_float = x_quant / self.scale
        return output_float

def load_models_for_evaluation(model_class, num_classes, q_value, weight_path):
    """
    一个辅助函数，用于加载两个核心模型：
    1. 原始的浮点模型
    2. 基于浮点模型创建的量化模型
    """
    # --- 加载浮点模型 (回答您的第一个问题) ---
    float_model = model_class(num_modes=num_classes)
    try:
        # 加载您通过 train.py 训练好的权重
        float_model.load_state_dict(torch.load(weight_path))
        print(f"成功从 '{weight_path}' 加载浮点模型权重。")
    except FileNotFoundError:
        print(f"错误: 找不到权重文件 '{weight_path}'。请先运行 train.py 进行训练。")
        return None, None
    float_model.eval()

    # --- 创建并加载量化模型 (回答您的第二个问题) ---
    scale = 2.0 ** q_value
    quantized_model = QuantizedNet(float_model, q_value)
    
    # 将浮点权重转换为整数权重，并加载到量化模型中
    float_state_dict = float_model.state_dict()
    quantized_state_dict = quantized_model.state_dict()

    for name, param in float_state_dict.items():
        # 核心权重转换逻辑
        if 'bias' in name:
            # 对偏置(bias)使用 scale * scale 进行缩放
            quantized_param = (param * scale**2).round()
        else:
            # 对权重(weight)使用 scale 进行缩放
            quantized_param = (param * scale).round()
        
        quantized_state_dict[name].copy_(quantized_param)
    quantized_model.eval()
    print("已成功创建并加载Q值量化模型。")
    
    return float_model, quantized_model

# ==============================================================================
# 3. 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # --- 全局配置区 ---
    num_modes = 12  # <-- 修改这里来切换模型

    # 使用与量化时完全相同的参数
    Q_VALUE = 10
    # 根据您训练时的情况，设置正确的epoch数
    if num_modes <= 19:
        TRAIN_EPOCHS = 15  # <-- 修改这里来切换训练轮数
    elif num_modes <=40 and num_modes > 19:
        TRAIN_EPOCHS = 40
    
    # 指定测试数据集路径
    DATASET_FILE = f'datasets\{num_modes}modes_mode_decomposition_dataset_test_classification.npz'
    # ==============================================================================

    print("="*60)
    print(f"开始评估模型: {num_modes}modes (Q={Q_VALUE})")
    print("="*60)

    # --- 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 动态选择模型类并构建路径 ---
    ModelClass = forClassification_net
    MODE_TO_EVALUATE = f"{num_modes}modes"

    float_weight_path = f'model_pth\{num_modes}modes_model_pth\model_dict_{TRAIN_EPOCHS}.pth'

    # --- 加载模型 ---
    # model_fp32 就是您训练后的模型
    # model_quant 就是量化后的模型
    model_fp32, model_quant = load_models_for_evaluation(ModelClass, num_modes, Q_VALUE, float_weight_path)

    if model_fp32 is None:
        exit()

    model_fp32.to(device)
    model_quant.to(device)

    # --- 加载测试数据 ---
    try:
        test_data = LPModesClassificationDataset(DATASET_FILE)
        test_loader = DataLoader(test_data, batch_size=64)
        print(f"成功加载测试数据集: {DATASET_FILE}")
    except FileNotFoundError:
        print(f"错误: 找不到数据集文件 '{DATASET_FILE}'。")
        exit()

    # --- 初始化评估指标 ---
    loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数

    correct_fp32, correct_quant = 0, 0
    total_loss_fp32, total_loss_quant = 0.0, 0.0
    total_samples = 0
    
    # --- 开始在整个测试集上评估 ---
    print("\n--- 正在测试集上进行评估... ---")
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader): # 假设数据集返回 (图像, 标签)
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 1. 浮点模型评估
            output_fp32 = model_fp32(imgs)
            loss_fp32 = loss_fn(output_fp32, labels)
            predicted_fp32 = torch.argmax(output_fp32, 1)
            correct_fp32 += (predicted_fp32 == labels).sum().item()
            total_loss_fp32 += loss_fp32.item()

            # 2. 量化模型评估
            output_quant = model_quant(imgs)
            loss_quant = loss_fn(output_quant, labels)
            predicted_quant = torch.argmax(output_quant, 1)
            correct_quant += (predicted_quant == labels).sum().item()
            total_loss_quant += loss_quant.item()

            # 累加样本总数
            total_samples += labels.size(0)

    # --- 计算并打印最终结果 (修改为分类结果) ---
    num_batches = len(test_loader)
    
    # 计算准确率
    accuracy_fp32 = 100 * correct_fp32 / total_samples
    accuracy_quant = 100 * correct_quant / total_samples
    
    # 计算平均损失
    avg_loss_fp32 = total_loss_fp32 / num_batches
    avg_loss_quant = total_loss_quant / num_batches
    
    # 计算性能变化
    accuracy_drop = accuracy_fp32 - accuracy_quant
    loss_increase_percent = 100 * (avg_loss_quant - avg_loss_fp32) / (avg_loss_fp32 + 1e-9)

    print("\n" + "="*75)
    print(f"评估结果总结 ({num_modes} modes, Q={Q_VALUE})")
    print("="*75)
    print(f"{'Metric':<20} | {'Float32 Model (训练后)':<25} | {'Quantized Model (量化后)':<25}")
    print("-"*75)
    print(f"{'Accuracy (%)':<20} | {accuracy_fp32:<25.4f} | {accuracy_quant:<25.4f}")
    print(f"{'Average Loss':<20} | {avg_loss_fp32:<25.8f} | {avg_loss_quant:<25.8f}")
    print("-"*75)
    print(f"Accuracy drop due to quantization: {accuracy_drop:.4f}%")
    print(f"Cross-Entropy Loss increased by: {loss_increase_percent:.2f}%")
    print("="*75)