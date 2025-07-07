import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# 假设您的数据集加载代码在一个名为 lp_dataset.py 的文件中
from lp_dataset import LPModesDataset 

def img_quant_export_qformat(q_value, dataset_path, save_path):
    """
    将输入特征图依据Q值进行对称量化, 并将结果保存为16位的Hex格式txt文件。

    Args:
        q_value (int): 用于量化的Q值 (例如 12)。
        dataset_path (str): .npz 数据集文件的路径。
        save_path (str): 量化后文件的保存路径。
    """
    print(f"--- 开始使用 Q={q_value} 对输入数据进行量化 ---")
    
    # 1. 定义量化参数
    scale_factor = 2.0 ** q_value

    # 2. 导入测试数据
    try:
        test_data = LPModesDataset(dataset_path)
        loader = DataLoader(test_data, batch_size=1)
        print(f"成功从 '{dataset_path}' 加载数据集。")
    except FileNotFoundError:
        print(f"错误: 找不到数据集文件 '{dataset_path}'。请检查路径。")
        return

    # 3. 准备保存文件
    # 确保保存路径的目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建目录: {save_dir}")

    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"已删除旧文件: {save_path}")

    # 4. 循环处理并保存
    with open(save_path, "ab") as f:
        for i, data in enumerate(loader):
            img, target = data
            
            # ★★★ 核心量化步骤：应用与您模型完全相同的量化逻辑 ★★★
            img_q_int = (img * scale_factor).round()

            # 检查数值范围是否超出16位整数
            min_val, max_val = img_q_int.min(), img_q_int.max()
            if min_val < -32768 or max_val > 32767:
                print(f"警告: 第 {i} 张图片量化后范围 [{min_val}, {max_val}] 超出16位整数表示范围！")

            # 将Tensor转换为numpy数组，并展平为一列
            img_q_int_numpy = torch.reshape(img_q_int, (-1, 1)).numpy().astype(np.int16)
            
            # ★★★ 关键：保存为4位十六进制格式 (%04x) 以代表16位整数 ★★★
            # 02x 用于8位, 04x 用于16位, 08x 用于32位
            np.savetxt(f, img_q_int_numpy, fmt="%04x")
    
    print(f"--- 量化完成！所有输入数据已保存到: {save_path} ---")


# --- 使用示例 ---
if __name__ == '__main__':
    # --- 配置区 ---
    # 使用与您量化模型时相同的Q值
    Q_VALUE_FOR_INPUT = 10 
    
    for i in range(1, 41):
        # 根据模型的模态数量设置数据集文件名
        DATASET_FILE = f'{i}modes_mode_decomposition_dataset_test_regression.npz'
        
        # 定义输出文件的路径和名称
        OUTPUT_TXT_PATH = f"./txt/input_data_q{i}_int16.txt"
        
        # --- 执行导出 ---
        img_quant_export_qformat(
            q_value=Q_VALUE_FOR_INPUT,
            dataset_path=DATASET_FILE,
            save_path=OUTPUT_TXT_PATH
        )
# --- 完成 ---