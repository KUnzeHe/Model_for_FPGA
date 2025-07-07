# Model_for_FPGA 项目说明

## 一、项目简介

本项目旨在实现基于深度学习的多模光纤模式识别与分类，并为后续FPGA部署提供量化模型和权重。项目分为两个主要子模块：

- `Model_for_Classification`：用于多模光纤模式的分类任务（如12模、40模等）。
- `Model_for_MD`：用于多模光纤模式分解（Mode Decomposition）等任务，支持3/5/6模等多种模式。

每个子模块均包含数据集生成、模型训练、量化、评估、权重导出等完整流程。

---

## 二、环境配置

建议使用Anaconda创建独立环境，已提供`environment.yml`文件，包含所有依赖（如PyTorch、Numpy、Torchvision、Tensorboard等）。

**步骤如下：**

1. 安装Anaconda（略）。
2. 进入对应子模块的`env_requirements`目录，执行：

   ```bash
   conda env create -f environment.yml
   conda activate DNN_cuda
   ```

3. 若需GPU加速，请确保CUDA驱动与PyTorch版本兼容。

---

## 三、文件结构说明

```
Model_for_FPGA/
├── Model_for_Classification/   # 分类任务相关
│   ├── coe/                    # 量化后权重的coe文件（用于FPGA）
│   ├── datasets/               # 存放训练/测试数据集（.npz格式）
│   ├── env_requirements/       # 环境依赖文件
│   ├── image/                  # 可视化图片
│   ├── logs/                   # Tensorboard日志
│   ├── model_pth/              # 训练得到的模型权重
│   ├── txt/                    # 导出的量化输入数据等
│   ├── main.py                 # 主入口，训练模型
│   ├── train.py                # 训练流程
│   ├── model_forClassification.py # 分类模型结构
│   ├── quant_evalueate.py      # 量化模型评估
│   ├── quant_general.py        # 量化权重导出
│   ├── img_export_quant.py     # 输入图片量化导出
│   ├── lp_dataset.py           # 数据集定义
│   ├── training dataset generating.py # 数据集生成脚本
│   └── ...
├── Model_for_MD/               # 模式分解等任务
│   └── ...（结构类似）
└── README.md                   # 项目说明
```
---

## 四、数据集说明

- 数据集为`.npz`格式，包含 speckle 图像及对应标签。
- 分类任务：`X`为输入图片，`y`为类别标签。
- 分解任务：`X`为输入图片，`y_amp`/`y_phase`为幅值/相位标签。
- 可通过`training dataset generating.py`自动生成，或放置在`datasets/`目录下。

---

## 五、主要功能模块说明

### 1. 数据集生成

- 运行`training dataset generating.py`，可自动生成不同模态数的数据集。
- 支持分类与回归两种任务，参数可在脚本内调整。

### 2. 模型训练

- 入口为`main.py`，会自动根据类别数创建模型并训练。
- 训练参数（类别数、batch size、epoch等）可在`main.py`或`train.py`中修改。
- 训练完成后，权重保存在`model_pth/`目录。

### 3. 量化与评估

- `quant_general.py`：将训练好的浮点模型权重量化为定点整数，并导出权重文件（支持FPGA部署）。
- `quant_evalueate.py`：对量化模型进行精度评估，与浮点模型对比，输出准确率、损失等指标。

### 4. 输入数据量化导出

- `img_export_quant.py`：将输入图片按Q值量化，导出为16位整数txt文件，便于硬件端测试。

### 5. COE文件生成

- 量化权重可进一步转换为COE文件，存放于`coe/`目录，用于FPGA初始化。

---

## 六、如何运行整个工程

以`Model_for_Classification`为例，流程如下：

### 1. 环境准备

```bash
cd Model_for_Classification/env_requirements
conda env create -f environment.yml
conda activate DNN_cuda
cd ..
```

### 2. 数据集生成（如需）

```bash
python "training dataset generating.py"
```
生成的数据集会自动保存在`datasets/`目录。

### 3. 模型训练

```bash
python main.py
```
- 训练参数可在`main.py`中调整（如类别数、epoch等）。
- 训练日志可用Tensorboard查看：

  ```bash
  tensorboard --logdir=logs
  ```

### 4. 量化权重导出

```bash
python quant_general.py
```
- 会在`model_pth/`下生成量化后的权重文件。

### 5. 量化模型评估

```bash
python quant_evalueate.py
```
- 输出量化前后模型的准确率、损失等对比。

### 6. 输入图片量化导出

```bash
python img_export_quant.py
```
- 生成的txt文件可用于FPGA端输入测试。

---

## 七、常见问题与建议

1. **数据集缺失/格式不符**：请先运行数据集生成脚本，或确保`datasets/`下有对应`.npz`文件。
2. **CUDA不可用**：如无GPU，可自动切换为CPU运行，但速度较慢。
3. **依赖安装失败**：建议使用Anaconda环境，避免包冲突。
4. **FPGA部署**：量化权重和输入数据均已按FPGA需求导出，COE文件在`coe/`目录。

---

## 八、参考与致谢

- 本项目参考了多模光纤模式识别相关论文与开源实现。
- 如有问题欢迎提Issue或联系作者。

---

如需针对`Model_for_MD`模块，流程与上述类似，仅需切换到对应目录操作即可。