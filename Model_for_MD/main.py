import warnings
warnings.filterwarnings('ignore')

from train import model_train
from model_for3modes import for3modes_net
from model_for5modes import for5modes_net
from model_for6modes import for6modes_net



if __name__ == "__main__":
    # 由于采用的是训练后静态量化方案,因此训练时无需例化插入量化节点的模型
    model_3modes = for3modes_net()
    model_5modes = for5modes_net()
    model_6modes = for6modes_net()

    # # 训练模型, 训练完成后保存为pth文件, 无需重新训练时建议注释掉
    turns = 2500  # 训练轮数
    model_train(network_class=model_3modes, out_dict_dir="model_pth/3modes", batch_size= 256, lr = 0.0006, epoch = 1500, n=1500)
    model_train(network_class=model_5modes, out_dict_dir="model_pth/5modes", batch_size= 512, lr = 0.001, epoch = 1000, n=1000)
    model_train(network_class=model_6modes, out_dict_dir="model_pth/6modes", batch_size= 512, lr = 0.001, epoch = 1000, n=1000)
    
    # # 训练完成后, 直接加载pth文件即可