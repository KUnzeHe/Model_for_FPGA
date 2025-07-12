import torch
from torch import nn
import copy
import os
import numpy as np

print("当前工作目录：", os.getcwd())

# ==========================================================================
# 步骤 1: 配置区 
# ==========================================================================
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
num_modes = 10  # <-- 修改这里来切换模型

# 根据论文或实验，为不同模型选择合适的Q值
# 这是一个可以调整的超参数
Q_VALUE = 10

if num_modes <= 19:
    WEIGHT_FILE_NAME = f'Model_for_Classification/model_pth/{num_modes}modes/model_dict_15.pth'
if num_modes <= 40 and num_modes > 19:
    WEIGHT_FILE_NAME = f'Model_for_Classification/model_pth/{num_modes}modes/model_dict_40.pth'

# ------------------------------------------------------------------------------
# 2. 定义一个通用的模拟定点运算的新模型类
#    这个类与之前的完全相同，只是名字改得更通用了
# ------------------------------------------------------------------------------
class QuantizedNet(nn.Module):
    def __init__(self, float_model, q_value):
        super().__init__()
        # 复制原始模型的结构
        self.features = copy.deepcopy(float_model.features)
        self.linear = copy.deepcopy(float_model.linear)
        
        # 定义量化参数
        self.q_value = q_value
        self.scale = 2.0 ** q_value
        
        self.float()

    def forward(self, x):
        # a. 输入量化
        x_quant = (x * self.scale).round()

        # b. 特征提取层
        for layer in self.features:
            x_quant = layer(x_quant)
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                x_quant = (x_quant / self.scale).round()

        # d. 全连接层
        for layer in self.linear:
            x_quant = layer(x_quant)
            if isinstance(layer, nn.Linear):
                x_quant = (x_quant / self.scale).round()
        
        # e. 输出反量化
        output_float = x_quant / self.scale
        
        return output_float

# ==========================================================================
# COE导出
# ==========================================================================
def tensor_to_coe(tensor, out_path, radix=16, dtype='int8'):
    tensor = tensor.flatten().cpu().numpy()
    if dtype == 'int8':
        tensor = tensor.astype(np.int8)
        tensor_u = tensor.astype(np.uint8)
        fmt = '{:02X}'
    elif dtype == 'int16':
        tensor = tensor.astype(np.int16)
        tensor_u = tensor.astype(np.uint16)
        fmt = '{:04X}'
    elif dtype == 'int32':
        tensor = tensor.astype(np.int32)
        tensor_u = tensor.astype(np.uint32)
        fmt = '{:08X}'
    else:
        raise ValueError('Unsupported dtype: ' + dtype + '. Supported: int8, int16, int32')
    with open(out_path, 'w') as f:
        f.write(f"MEMORY_INITIALIZATION_RADIX={radix};\n")
        f.write("MEMORY_INITIALIZATION_VECTOR=\n")
        for i, val in enumerate(tensor_u):
            if i == len(tensor_u) - 1:
                f.write(fmt.format(val) + ';\n')
            else:
                f.write(fmt.format(val) + ',\n')

def export_coe_files(weight_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    layer_mapping = {
        'features.0.weight': 'conv1.weight_int8.coe',
        'features.0.bias': 'conv1.bias_int32.coe',
        'features.2.weight': 'conv2.weight_int8.coe',
        'features.2.bias': 'conv2.bias_int32.coe',
        'features.6.weight': 'conv3.weight_int8.coe',
        'features.6.bias': 'conv3.bias_int32.coe',
        'linear.1.weight': 'linear1.weight_int8.coe',
        'linear.1.bias': 'linear1.bias_int32.coe',
        'linear.3.weight': 'linear2.weight_int8.coe',
        'linear.3.bias': 'linear2.bias_int32.coe',
    }
    exported_count = 0
    for key, tensor in weight_dict.items():
        if key in layer_mapping:
            out_path = os.path.join(out_dir, layer_mapping[key])
            if 'weight' in key:
                dtype = 'int8'
            elif 'bias' in key:
                dtype = 'int32'
            else:
                continue
            print(f"导出 {key} -> {layer_mapping[key]}")
            tensor_to_coe(tensor, out_path, radix=16, dtype=dtype)
            exported_count += 1
        else:
            print(f"警告: 未找到层 {key} 的映射，跳过")
    print(f"\n成功导出 {exported_count} 个COE文件到 {out_dir}")

# --- 主逻辑开始 ---
if __name__ == '__main__':

    # ==========================================================================
    # 步骤 3: 根据配置动态加载模型和设置路径
    # ==========================================================================
    print(f"--- 当前配置: 模型={num_modes}modes, Q值={Q_VALUE} ---")
    
    from model_forClassification import forClassification_net as ModelClass
    # 根据 num_modes 动态设置输入形状
    input_shape = (1, 1, 16, 16)
 
   
    # 构建文件路径
    weight_path = os.path.join('model_pth', f'{num_modes}modes', WEIGHT_FILE_NAME)
    output_quantized_weight_path = f"Model_for_Classification/model_pth/{num_modes}modes_model_pth/for{num_modes}modes_quantized_int_weights.pth"
    
    SCALE = 2.0 ** Q_VALUE
    print(f"使用的Q值为: {Q_VALUE}, 对应的缩放因子 Scale 为: {SCALE}")

    # ==========================================================================
    # 步骤 4: 加载预训练的浮点模型 (代码与之前基本一致)
    # ==========================================================================
    print("\n--- 正在加载浮点模型 ---")
    float_model = ModelClass(num_modes)
    
    try:
        float_model.load_state_dict(torch.load(weight_path))
        print(f"成功从 '{weight_path}' 加载预训练权重。")
    except FileNotFoundError:
        print(f"警告: 未找到权重文件 '{weight_path}'。将使用随机初始化的权重进行演示。")
    
    float_model.eval()

    # ==========================================================================
    # 步骤 5: 创建量化模型并执行权重转换 (代码与之前完全一致)
    # ==========================================================================
    print("\n--- 正在创建量化模型并转换权重 ---")
    
    quantized_model = QuantizedNet(float_model, Q_VALUE)
    
    float_state_dict = float_model.state_dict()
    quantized_state_dict = quantized_model.state_dict()

    for name, param in float_state_dict.items():

        if 'bias' in name:
            # 对偏置(bias)使用 SCALE * SCALE 进行缩放
            quantized_param = (param * SCALE**2).round()
        else:
            # 对权重(weight)使用 SCALE 进行缩放
            quantized_param = (param * SCALE).round()
        
        min_val, max_val = quantized_param.min(), quantized_param.max()
        if min_val < -32768 or max_val > 32767:
            print(f"警告: 参数 '{name}' 的量化值范围 [{min_val}, {max_val}] 超出16位整数表示范围！")
            
        quantized_state_dict[name].copy_(quantized_param)
    
    quantized_model.eval()
    print("权重转换完成！")
    
    # ==========================================================================
    # 步骤 6: 验证和对比 (代码与之前基本一致)
    # ==========================================================================
    print("\n--- 正在验证模型输出 ---")
    
    # 使用一个随机输入作为示例
    dummy_input = torch.randn(*input_shape)
    
    # 在不计算梯度的模式下执行
    with torch.no_grad():
        # 获取浮点模型的原始输出 (logits)
        float_output = float_model(dummy_input)
        # 获取量化模型的原始输出 (logits)
        quantized_output = quantized_model(dummy_input)
        
    # --- 核心修改部分 ---
    # 对于分类任务，我们更关心的是最终的预测类别是否一致
    # 通过 argmax 获取预测的类别索引
    float_prediction = torch.argmax(float_output, dim=1)
    quantized_prediction = torch.argmax(quantized_output, dim=1)

    print(f"\n测试输入形状: {dummy_input.shape}")
    print(f"浮点模型原始输出 (logits): {float_output}")
    print(f"量化模型原始输出 (logits): {quantized_output}")
    print("-" * 50)
    print("分类决策对比:")
    print(f"  -> 浮点模型的预测类别: {float_prediction.item()}")
    print(f"  -> 量化模型的预测类别: {quantized_prediction.item()}")
    print("-" * 50)

    # 检查预测结果是否一致
    if float_prediction.item() == quantized_prediction.item():
        print("预测结果一致！量化模型保持了与浮点模型相同的分类决策。")
    else:
        print("预测结果不一致！量化对该特定输入的分类决策产生了影响。")

    # 同时，我们仍然可以计算均方误差(MSE)来衡量输出logits的数值差异
    # 这可以作为衡量量化精度的指标之一
    mse_loss = nn.MSELoss()
    quantization_error = mse_loss(float_output, quantized_output)
    print(f"\n量化引入的输出Logits均方误差 (MSE): {quantization_error.item():.8f}")


    # ==========================================================================
    # 保存量化后的整数权重 (代码与之前基本一致)
    # ==========================================================================
    integer_weights = {name: param.int() for name, param in quantized_model.state_dict().items()}
    torch.save(integer_weights, output_quantized_weight_path)
    print(f"\n已将量化后的整数权重保存到 {output_quantized_weight_path}")

    # === COE导出 ===
    coe_output_dir = f"Model_for_Classification/coe/coe_{num_modes}modes"
    print(f"\n--- 正在导出COE文件 ---")
    export_coe_files(integer_weights, coe_output_dir)
    print(f"\nCOE文件已导出到: {coe_output_dir}")