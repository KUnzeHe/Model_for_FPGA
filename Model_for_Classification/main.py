import warnings
warnings.filterwarnings('ignore')

from train import model_train
from model_forClassification import forClassification_net


if __name__ == "__main__":
    # 由于采用的是训练后静态量化方案,因此训练时无需例化插入量化节点的模型
    num_modes = 12  # 假设有10个类别
    model_forClassification = forClassification_net(num_modes)  # 假设有10个类别

    # # 训练模型, 训练完成后保存为pth文件, 无需重新训练时建议注释掉
    if num_modes <= 19:
        model_train(network_class=model_forClassification, out_dict_dir=f"model_pth/{num_modes}modes_model_pth", batch_size= 64, lr = 0.0001, epoch = 15, n=15)
    if num_modes <=40 and num_modes > 19:
        model_train(network_class=model_forClassification, out_dict_dir=f"model_pth/{num_modes}modes_model_pth", batch_size= 128, lr = 0.0001, epoch = 40, n=40)
    
