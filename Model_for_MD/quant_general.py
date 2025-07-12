import torch
from torch import nn
import copy
import os
import numpy as np

# ==========================================================================
# 步骤 1: 配置区 - 您只需要修改这里！
# ==========================================================================
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 在这里选择要量化的模型: '3modes', '5modes', 或 '6modes' ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
MODE_TYPE = '6modes'  # <-- 修改这里来切换模型

# 根据论文或实验，为不同模型选择合适的Q值
# 这是一个可以调整的超参数
Q_VALUE = 12

# 指定预训练权重文件的名称
if MODE_TYPE == '3modes':
    WEIGHT_FILE_NAME = 'model/3modes/model_dict_1000.pth'
elif MODE_TYPE == '5modes':
    WEIGHT_FILE_NAME = 'model/5modes/model_dict_2500.pth'
elif MODE_TYPE == '6modes':
    WEIGHT_FILE_NAME = 'model/6modes/model_dict_2500.pth' 
# ==========================================================================

# ==========================================================================
# COE导出功能
# ==========================================================================
def tensor_to_coe(tensor, out_path, radix=16, dtype='int8'):
    """
    将tensor导出为COE文件格式
    
    Args:
        tensor: torch.Tensor - 要导出的张量
        out_path: str - 输出的coe文件路径
        radix: int - 进制，默认16（十六进制）
        dtype: str - 数据类型，'int8', 'int16' 或 'int32'
    """
    tensor = tensor.flatten().cpu().numpy()
    
    if dtype == 'int8':
        tensor = tensor.astype(np.int8)
        tensor_u = tensor.astype(np.uint8)
        fmt = '{:02X}'  # 2位十六进制
    elif dtype == 'int16':
        tensor = tensor.astype(np.int16)
        tensor_u = tensor.astype(np.uint16)
        fmt = '{:04X}'  # 4位十六进制
    elif dtype == 'int32':
        tensor = tensor.astype(np.int32)
        tensor_u = tensor.astype(np.uint32)
        fmt = '{:08X}'  # 8位十六进制
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

def export_coe_files(weight_dict, out_dir, model_type):
    """
    导出所有层的权重和偏置为COE文件
    
    Args:
        weight_dict: dict - 包含权重和偏置的字典
        out_dir: str - 输出目录
        model_type: str - 模型类型 ('3modes', '5modes', '6modes')
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 定义层名映射（根据你的模型结构）
    # 3modes: features[0,3,6] = conv1,conv2,conv3; linear[1,3] = linear1,linear2
    # 5modes: features[0,3,6,9,12,15] = conv1,conv2,conv3,conv4,conv5,conv6; linear[1,3,5] = linear1,linear2,linear3  
    # 6modes: features[0,3,6,9,12,15,18,21] = conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8; linear[1,3,5] = linear1,linear2,linear3
    
    layer_mapping = {
        'features.0.weight': 'conv1.weight_int8.coe',
        'features.0.bias': 'conv1.bias_int32.coe',
        'features.2.weight': 'conv2.weight_int8.coe',
        'features.2.bias': 'conv2.bias_int32.coe',
        'features.6.weight': 'conv3.weight_int8.coe',
        'features.6.bias': 'conv3.bias_int32.coe',
        'features.8.weight': 'conv4.weight_int8.coe',
        'features.8.bias': 'conv4.bias_int32.coe',
        'features.12.weight': 'conv5.weight_int8.coe',
        'features.12.bias': 'conv5.bias_int32.coe',
        'features.14.weight': 'conv6.weight_int8.coe',
        'features.14.bias': 'conv6.bias_int32.coe',
        'features.18.weight': 'conv7.weight_int8.coe',
        'features.18.bias': 'conv7.bias_int32.coe',
        'features.20.weight': 'conv8.weight_int8.coe',
        'features.20.bias': 'conv8.bias_int32.coe',
        'linear.1.weight': 'linear1.weight_int8.coe',
        'linear.1.bias': 'linear1.bias_int32.coe',
        'linear.3.weight': 'linear2.weight_int8.coe',
        'linear.3.bias': 'linear2.bias_int32.coe',
        'linear.5.weight': 'linear3.weight_int8.coe',
        'linear.5.bias': 'linear3.bias_int32.coe',
    }
    
    exported_count = 0
    for key, tensor in weight_dict.items():
        if key in layer_mapping:
            out_path = os.path.join(out_dir, layer_mapping[key])
            
            # 确定数据类型
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

# --- 主逻辑开始 ---
if __name__ == '__main__':

    # ==========================================================================
    # 步骤 3: 根据配置动态加载模型和设置路径
    # ==========================================================================
    print(f"--- 当前配置: 模型={MODE_TYPE}, Q值={Q_VALUE} ---")
    
    if MODE_TYPE == '3modes':
        from model_for3modes import for3modes_net as ModelClass
        input_shape = (1, 1, 16, 16)
    elif MODE_TYPE == '5modes':
        from model_for5modes import for5modes_net as ModelClass
        input_shape = (1, 1, 32, 32)
    elif MODE_TYPE == '6modes':
        from model_for6modes import for6modes_net as ModelClass
        input_shape = (1, 1, 32, 32)
    else:
        raise ValueError("无效的 MODE_TYPE! 请选择 '3modes', '5modes', 或 '6modes'.")

    # 构建文件路径
    weight_path = os.path.join('model_pth', MODE_TYPE, WEIGHT_FILE_NAME)
    if MODE_TYPE == '3modes':
        output_quantized_weight_path = f"Model_for_MD/model_pth/3modes/for{MODE_TYPE}_quantized_int_weights.pth"
    elif MODE_TYPE == '5modes':
        output_quantized_weight_path = f"Model_for_MD/model_pth/5modes/for{MODE_TYPE}_quantized_int_weights.pth"
    elif MODE_TYPE == '6modes':
        output_quantized_weight_path = f"Model_for_MD/model_pth/6modes/for{MODE_TYPE}_quantized_int_weights.pth"
    
    print(f"预训练权重文件路径: {weight_path}")
    print(f"量化后的整数权重将保存到: {output_quantized_weight_path}")

    
    SCALE = 2.0 ** Q_VALUE
    print(f"使用的Q值为: {Q_VALUE}, 对应的缩放因子 Scale 为: {SCALE}")

    # ==========================================================================
    # 步骤 4: 加载预训练的浮点模型 (代码与之前基本一致)
    # ==========================================================================
    print("\n--- 正在加载浮点模型 ---")
    float_model = ModelClass()
    
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
    
    dummy_input = torch.randn(*input_shape)
    
    with torch.no_grad():
        float_output = float_model(dummy_input)
        quantized_output = quantized_model(dummy_input)
        
    print(f"\n测试输入形状: {dummy_input.shape}")
    print(f"浮点模型输出: {float_output}")
    print(f"量化模型输出: {quantized_output}")
    
    mse_loss = nn.MSELoss()
    quantization_error = mse_loss(float_output, quantized_output)
    print(f"\n量化引入的均方误差 (MSE): {quantization_error.item():.8f}")

    # ==========================================================================
    # 保存量化后的整数权重 (代码与之前基本一致)
    # ==========================================================================
    integer_weights = {name: param.int() for name, param in quantized_model.state_dict().items()}
    torch.save(integer_weights, output_quantized_weight_path)
    print(f"\n已将量化后的整数权重保存到 {output_quantized_weight_path}")
    
    # ==========================================================================
    # 步骤 7: 导出COE文件
    # ==========================================================================
    print("\n--- 正在导出COE文件 ---")
    
    # 构建COE输出目录
    coe_output_dir = f"Model_for_MD/coe/coe_{MODE_TYPE}"
    
    # 导出COE文件
    export_coe_files(integer_weights, coe_output_dir, MODE_TYPE)
    
    print(f"\n=== 量化完成！===")
    print(f"1. 量化权重已保存到: {output_quantized_weight_path}")
    print(f"2. COE文件已导出到: {coe_output_dir}")
    print(f"3. 量化误差 (MSE): {quantization_error.item():.8f}")

print("\n==== 权重/偏置 shape 检查 ====")
for key, tensor in integer_weights.items():
    print(f"{key}: {tensor.shape}")