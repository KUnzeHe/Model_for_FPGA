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
from model_for3modes import for3modes_net
from model_for5modes import for5modes_net
from model_for6modes import for6modes_net
from lp_dataset import LPModesDataset

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

def load_models_for_evaluation(model_class, q_value, weight_path):
    """
    一个辅助函数，用于加载两个核心模型：
    1. 原始的浮点模型
    2. 基于浮点模型创建的量化模型
    """
    # --- 加载浮点模型 (回答您的第一个问题) ---
    float_model = model_class()
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
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 请在这里选择你要评估的模型: '3modes', '5modes', 或 '6modes' ★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    MODE_TO_EVALUATE = '6modes'  # <-- 修改这里来切换模型

    # 使用与量化时完全相同的参数
    Q_VALUE = 12
    # 根据您训练时的情况，设置正确的epoch数
    TRAIN_EPOCHS = 2500  # <-- 修改这里来切换训练轮数
    
    # 指定测试数据集路径
    if MODE_TO_EVALUATE == '3modes':
        DATASET_FILE = 'Model_for_MD/datasets/3modes_mode_decomposition_dataset_test_regression.npz'
    elif MODE_TO_EVALUATE == '5modes':
        DATASET_FILE = 'Model_for_MD/datasets/5modes_mode_decomposition_dataset_test_regression.npz'
    elif MODE_TO_EVALUATE == '6modes':
        DATASET_FILE = 'Model_for_MD/datasets/6modes_mode_decomposition_dataset_test_regression.npz'
    # ==============================================================================

    print("="*60)
    print(f"开始评估模型: {MODE_TO_EVALUATE} (Q={Q_VALUE})")
    print("="*60)

    # --- 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 动态选择模型类并构建路径 ---
    if MODE_TO_EVALUATE == '3modes': ModelClass = for3modes_net
    elif MODE_TO_EVALUATE == '5modes': ModelClass = for5modes_net
    elif MODE_TO_EVALUATE == '6modes': ModelClass = for6modes_net
    else: raise ValueError("无效的 MODE_TO_EVALUATE!")

    float_weight_path = os.path.join('model_pth', MODE_TO_EVALUATE, f'model_dict_{TRAIN_EPOCHS}.pth')

    # --- 加载模型 ---
    # model_fp32 就是您训练后的模型
    # model_quant 就是量化后的模型
    model_fp32, model_quant = load_models_for_evaluation(ModelClass, Q_VALUE, float_weight_path)

    if model_fp32 is None:
        exit()

    model_fp32.to(device)
    model_quant.to(device)

    # --- 加载测试数据 ---
    try:
        test_data = LPModesDataset(DATASET_FILE)
        test_loader = DataLoader(test_data, batch_size=64)
        print(f"成功加载测试数据集: {DATASET_FILE}")
    except FileNotFoundError:
        print(f"错误: 找不到数据集文件 '{DATASET_FILE}'。")
        exit()

    # --- 初始化评估指标 ---
    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss() 

    total_mse_fp32, total_mae_fp32 = 0.0, 0.0
    total_mse_quant, total_mae_quant = 0.0, 0.0
    
    # --- 开始在整个测试集上评估 ---
    print("\n--- 正在测试集上进行评估... ---")
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(test_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 1. 获取浮点模型的输出 (回答问题1：训练表现)
            output_fp32 = model_fp32(imgs)

            # 2. 获取量化模型的输出 (回答问题2：量化表现)
            output_quant = model_quant(imgs)

            # 累加损失
            total_mse_fp32 += mse_loss_fn(output_fp32, targets).item()
            total_mae_fp32 += mae_loss_fn(output_fp32, targets).item()

            total_mse_quant += mse_loss_fn(output_quant, targets).item()
            total_mae_quant += mae_loss_fn(output_quant, targets).item()

    # --- 计算并打印最终结果 ---
    num_batches = len(test_loader)
    avg_mse_fp32 = total_mse_fp32 / num_batches
    avg_mae_fp32 = total_mae_fp32 / num_batches
    avg_mse_quant = total_mse_quant / num_batches
    avg_mae_quant = total_mae_quant / num_batches
    
    # 计算性能下降的百分比
    mse_increase_percent = 100 * (avg_mse_quant - avg_mse_fp32) / (avg_mse_fp32 + 1e-9)
    mae_increase_percent = 100 * (avg_mae_quant - avg_mae_fp32) / (avg_mae_fp32 + 1e-9)

    print("\n" + "="*70)
    print(f"评估结果总结 ({MODE_TO_EVALUATE}, Q={Q_VALUE})")
    print("="*70)
    print(f"{'Metric':<15} | {'Float32 Model (训练后)':<25} | {'Quantized Model (量化后)':<25}")
    print("-"*70)
    print(f"{'Average MSE':<15} | {avg_mse_fp32:<25.8f} | {avg_mse_quant:<25.8f}")
    print(f"{'Average MAE':<15} | {avg_mae_fp32:<25.8f} | {avg_mae_quant:<25.8f}")
    print("-"*70)
    print(f"MSE due to quantization increased by: {mse_increase_percent:.2f}%")
    print(f"MAE due to quantization increased by: {mae_increase_percent:.2f}%")
    print("="*70)