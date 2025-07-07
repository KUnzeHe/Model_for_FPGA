# 使用经过定点化后的模型进行相关实验测试
# 作为Verilog/FPGA/ASIC计算结果的参考
# 这一过程亦可在MATLAB中进行,但MATLAB不方便进行量化模型的模拟
# 运行此文件前请先运行main获得各类参数文件

# 禁止弹出警告
import warnings
warnings.filterwarnings('ignore')

# 导入相关包
import torchvision.datasets
import torch
from torch.utils.data import DataLoader

# 导入相关模块
from nn_quant_basic import *

from lp_dataset import LPModesDataset

# 设置device：fp32用GPU，int8用CPU
device_fp32 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_int8 = torch.device("cpu")

# 比较两个模型结果之间的差距,打印相关信息
def error_calculate(res1, res2, name):
    print("________________________{}_res____________________________".format(name))
    # 需要查看中间结果时, 取消注释下面两行, 默认只查看错误信息
    # print("res1:{}\nres2:{}".format(res1, res2))
    # print("error:{}".format((res1 - res2).to(torch.int8)))
    print("error_times:{}".format((res1 != res2).sum()))
    print("average error:{}".format(torch.mean(abs((res1 - res2).to(torch.int8).to(torch.float)))))

# 设置量化后端
torch.backends.quantized.engine = 'fbgemm'

# 量化后模型的加载（int8只能用CPU）
model_int8_base = cifar10_net(is_quant=True)
model_int8_base = model_int8_base.to(device_int8)
state_dict = torch.load("./model_pth/model_int8.pth")
model_int8_base.qconfig = torch.quantization.get_default_qconfig('x86')  
load_model_prepared = torch.quantization.prepare(model_int8_base)
model_int8 = torch.quantization.convert(load_model_prepared)
model_int8.load_state_dict(state_dict)
model_int8 = model_int8.to(device_int8)
model_int8.eval()

# 浮点模型加载（用GPU）
model_fp32 = torch.load("./model_pth/model_2500.pth", weights_only=False)
model_fp32 = model_fp32.to(device_fp32)
model_fp32.eval()

# LP modes测试集
batch_size = 1
test_data = LPModesDataset('mode_decomposition_dataset_test_regression.npz')
test_loader = DataLoader(test_data, batch_size=batch_size)

# 单张图片输入测试
# # 对量化模型model_int8,输入测试图片,作为自定义scale量化模型的对照
i = 1
for data in test_loader:
    if i == 423:
        img, target = data

        img_cpu = img.to(device_int8)    # int8模型用CPU
        img_gpu = img.to(device_fp32)    # fp32模型用GPU

        model_int8(img_cpu)
        break
    i = i + 1
n = 16
q_net_out = model_int8.linear2_res.int_repr()  # model_int8第linear2层的输出

# 从txt读取量化后的img作为定点量化模型的输入
img_path = "./txt/img_uint8.txt"  # 文件路径
str_list = txt_hex_to_dec_list(img_path)  # 将激活文件转换为十进制字符列表
# 从txt读取量化后测试数据集第n张图片作为定点量化模型的输入
img = read_img_from_str_list(int_str_list=str_list, n=769, img_channel=1, img_size_h=16, img_size_w=16)
img = img.to(device_int8)

# 实例化定点量化模型,输入img,得出输出结果
# 注意这里的定点数小数位n需要与nn_quant_export.py中的保持一致, 需要修改时需要先修改nn_quant_export.py,并运行生成新的fix_scale字典
model_int8_fix = q_cifar10(n=16, is_get_intermediate=True)
fix_net_out = model_int8_fix.forward_q(img)

# 读取定点量化模型model_int8_fix的中间结果,并与量化模型model_int8的中间结果作为对比, 打印相关误差信息, 不需要时注释掉
error_calculate(model_int8_fix.conv1_res, model_int8.conv1_res.int_repr(), name="conv1")
error_calculate(model_int8_fix.relu1_res, model_int8.relu1_res.int_repr(), name="relu1")
error_calculate(model_int8_fix.maxpool1_res, model_int8.maxpool1_res.int_repr(), name="maxpooling1")
error_calculate(model_int8_fix.conv2_res, model_int8.conv2_res.int_repr(), name="conv2")
error_calculate(model_int8_fix.relu2_res, model_int8.relu2_res.int_repr(), name="relu2")
error_calculate(model_int8_fix.maxpool2_res, model_int8.maxpool2_res.int_repr(), name="maxpooling2")
error_calculate(model_int8_fix.conv3_res, model_int8.conv3_res.int_repr(), name="conv3")
error_calculate(model_int8_fix.relu3_res, model_int8.relu3_res.int_repr(), name="relu1")
error_calculate(model_int8_fix.maxpool3_res, model_int8.maxpool3_res.int_repr(), name="maxpooling3")
error_calculate(model_int8_fix.flatten_res, model_int8.flatten_res.int_repr(), name="flatten")
error_calculate(model_int8_fix.linear1_res, model_int8.linear1_res.int_repr(), name="linear1")
error_calculate(model_int8_fix.linear2_res, model_int8.linear2_res.int_repr(), name="linear2")

# 单张图片测试结果总结
print("model_int8_fix_target:{}".format(fix_net_out.argmax(1)))
print("model_int8_target:{}".format(q_net_out.argmax(1)))
print("model_int8_fix_out:{}".format(fix_net_out))
print("model_int8_out:    {}".format(q_net_out))

# 用测试数据集比较量化模型、定点量化模型、浮点模型的回归性能
i = 1
sum_mse_fp32 = 0.0
sum_mse_int8 = 0.0
sum_mse_fix = 0.0
sum_mae_fp32 = 0.0
sum_mae_int8 = 0.0
sum_mae_fix = 0.0
total_test_times = len(test_data)

for data in test_loader:
    img, target = data  # target shape: (1, 19)
    img_cpu = img.to(device_int8)
    img_gpu = img.to(device_fp32)
    img_q = read_img_from_str_list(str_list, n=i, img_channel=1, img_size_w=16,
                                   img_size_h=16)
    img_q = img_q.to(device_int8)
    int8_out = model_int8(img_cpu)
    fp32_out = model_fp32(img_gpu)

    fix_out = model_int8_fix.forward_q(img_q)
    fix_scale = quant_dict['linear2.out.scale']
    fix_zero_point = quant_dict['linear2.out.zero_point']
    fix_out_float = (fix_out.float() - fix_zero_point) * fix_scale


    # 如果模型输出是量化的（如int_repr），需要dequantize或转为float
    if hasattr(int8_out, "dequantize"):
        int8_out = int8_out.dequantize()
    if hasattr(fix_out_float, "dequantize"):
        fix_out_float = fix_out_float.dequantize()
    # 保证所有输出和target都是float类型
    fp32_out = fp32_out.cpu().float().squeeze()
    int8_out = int8_out.cpu().float().squeeze()
    fix_out_float = fix_out_float.cpu().float().squeeze()
    target = target.cpu().float().squeeze()

    # 计算MSE和MAE
    mse_fp32 = ((fp32_out - target) ** 2).mean().item()
    mse_int8 = ((int8_out - target) ** 2).mean().item()
    mse_fix = ((fix_out_float - target) ** 2).mean().item()
    mae_fp32 = (fp32_out - target).abs().mean().item()
    mae_int8 = (int8_out - target).abs().mean().item()
    mae_fix = (fix_out_float - target).abs().mean().item()

    sum_mse_fp32 += mse_fp32
    sum_mse_int8 += mse_int8
    sum_mse_fix += mse_fix
    sum_mae_fp32 += mae_fp32
    sum_mae_int8 += mae_int8
    sum_mae_fix += mae_fix

    if i % 1000 == 0:
        print("第{}轮测试完成".format(i))
        print("已输入{}张测试图片".format(batch_size * i))
        print("当前浮点模型MSE: {:.6f}, MAE: {:.6f}".format(mse_fp32, mae_fp32))
        print("当前量化模型MSE: {:.6f}, MAE: {:.6f}".format(mse_int8, mae_int8))
        print("当前定点量化模型MSE: {:.6f}, MAE: {:.6f}".format(mse_fix, mae_fix))
    i += 1

print("========全部测试集回归指标========")
print("浮点模型 平均MSE: {:.6f}, 平均MAE: {:.6f}".format(sum_mse_fp32 / total_test_times, sum_mae_fp32 / total_test_times))
print("量化模型 平均MSE: {:.6f}, 平均MAE: {:.6f}".format(sum_mse_int8 / total_test_times, sum_mae_int8 / total_test_times))
print("定点量化模型 平均MSE: {:.6f}, 平均MAE: {:.6f}".format(sum_mse_fix / total_test_times, sum_mae_fix / total_test_times))
print("浮点->量化 MSE上升: {:.2f}%".format(
    100 * (sum_mse_int8 - sum_mse_fp32) / (sum_mse_fp32 + 1e-9)))
print("量化->定点 MSE上升: {:.2f}%".format(
    100 * (sum_mse_fix - sum_mse_int8) / (sum_mse_int8 + 1e-9)))
print("浮点->定点 MSE上升: {:.2f}%".format(
    100 * (sum_mse_fix - sum_mse_fp32) / (sum_mse_fp32 + 1e-9)))