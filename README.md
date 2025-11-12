# 智能商品分类预测系统（基于BERT）

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)


## 目录
- [项目简介](#项目简介)
- [核心能力](#核心能力)
- [环境搭建](#环境搭建)
- [快速上手](#快速上手)
- [项目结构](#项目结构)
- [使用指南](#使用指南)
- [模型优化方案](#模型优化方案)
- [部署与接口测试](#部署与接口测试)
- [数据集说明](#数据集说明)


## 项目简介
在电商场景中，商品上架需手动填写品牌、品类等信息，流程繁琐且效率低下。本系统以**BERT预训练模型**为核心，通过分析商品标题文本，自动预测商品所属品类，帮助商家简化商品发布流程，提升上架效率，降低人工操作成本。


## 核心能力
1. 自动品类预测：输入商品标题，实时输出精准分类结果
2. 全链路数据处理：支持原始数据清洗、标签标准化、文本分词编码
3. 多维度性能评估：覆盖准确率、精确率、召回率、F1分数四大核心指标
4. 高效训练管控：集成早停、混合精度训练、断点续训，平衡训练效率与模型效果
5. 灵活使用方式：提供命令行交互式预测与HTTP API服务两种部署形态
6. 模块化架构：代码分层清晰，支持功能扩展与二次开发


## 环境搭建
### 1. 创建虚拟环境
通过Conda创建独立环境，避免依赖冲突：
```bash
# 创建环境（指定Python 3.12）
conda create -n product-classify python=3.12

# 激活环境
conda activate product-classify
```

### 2. 安装依赖库
先通过`nvidia-smi`查看本地CUDA版本，再安装对应版本的PyTorch及其他依赖：
```bash
# 安装PyTorch（以CUDA 12.6为例，需根据实际CUDA版本调整）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装其他核心依赖（transformers、FastAPI等）
pip install transformers datasets scikit-learn tensorboard tqdm jupyter fastapi uvicorn
```


## 快速上手
通过`src/main.py`入口脚本执行各类操作，命令格式统一为：
```bash
python src/main.py [操作类型]
```

| 操作类型 | 执行命令                          | 功能描述                                  |
|----------|-----------------------------------|-------------------------------------------|
| 数据预处理 | `python src/main.py process`      | 清洗原始数据、处理标签、分词编码并保存    |
| 模型训练   | `python src/main.py train`        | 启动模型训练，自动记录日志并保存最优模型  |
| 模型评估   | `python src/main.py evaluate`     | 基于测试集计算模型性能指标                |
| 交互式预测 | `python src/main.py predict`      | 命令行输入商品标题，实时查看分类结果      |
| API服务部署 | `python src/main.py serve`        | 启动FastAPI服务，提供HTTP预测接口         |


## 项目结构
```
product-classify_bert/
├── data/                # 数据存储目录
│   ├── raw/             # 原始数据集（train.txt/test.txt/valid.txt）
│   └── processed/       # 预处理后的数据（供模型直接使用）
├── logs/                # TensorBoard训练日志（可视化训练过程）
├── models/              # 模型文件目录
│   ├── model.pt         # 最优模型权重
│   └── checkpoint.pt    # 训练断点文件
├── pretrained/          # 预训练模型目录（存放bert-base-chinese）
└── src/                 # 核心源码目录
    ├── configuration/   # 项目配置文件（路径、超参数等）
    ├── model/           # 模型定义（BERT+分类头）
    ├── preprocess/      # 数据预处理脚本（加载、清洗、分词）
    ├── runner/          # 核心逻辑（训练、预测、评估）
    ├── web/             # API服务代码（FastAPI路由、Schema）
    └── main.py          # 程序入口（支持多操作类型）
```


## 使用指南
### 1. 数据预处理
- 前提：将原始数据集（train.txt、test.txt、valid.txt）放入`data/raw/`目录，数据格式为`label\ttext_a`（制表符分隔）
- 执行命令后，预处理后的数据会保存到`data/processed/`，自动完成：
  - 过滤空值样本（text_a或label为None）
  - 标签标准化（转换为ClassLabel类型）
  - BERT分词（生成input_ids、attention_mask）

### 2. 模型训练
- 训练过程中，TensorBoard日志会保存到`logs/[时间戳]/`，可通过以下命令查看：
  ```bash
  tensorboard --logdir=logs/[时间戳]
  ```
- 自动保存**最优模型**（基于验证集损失）到`models/model.pt`，支持断点续训（检测到`models/checkpoint.pt`时自动恢复）

### 3. 交互式预测
- 执行命令后，输入商品标题（如“240ML*15养元2430六个核桃”），系统会输出：
  ```
  预测类别ID: 2，类别名称: 酒饮冲调
  ```
- 输入`q`或`quit`可退出交互模式

### 4. API服务部署
- 服务启动后，默认监听`0.0.0.0:8000`，可通过以下地址访问接口文档：
  ```
  http://localhost:8000/docs
  ```
- 接口请求示例（POST `/predict`）：
  ```json
  {
    "text": "911-267遥控车"
  }
  ```
- 接口响应示例：
  ```json
  {
    "text": "911-267遥控车",
    "pred_id": 4,
    "pred_label": "玩具"
  }
  ```


## 模型优化方案
### 1. 早停机制（Early Stopping）
- 当验证集损失连续2轮未下降时，自动终止训练，避免过拟合
- 实时保存最优模型，无需手动干预

### 2. 混合精度训练（Mixed Precision）
- 结合`torch.float16`（半精度）与`torch.float32`（单精度）：
  - 卷积、矩阵乘法等操作使用半精度，提升速度、减少显存占用
  - 归一化、损失计算等操作使用单精度，保证数值稳定性
- 通过`torch.autocast`和`GradScaler`实现自动精度切换

### 3. 断点续训（Checkpoint）
- 训练过程中定期保存：
  - 模型权重（model.state_dict()）
  - 优化器状态（optimizer.state_dict()）
  - 梯度缩放器状态（scaler.state_dict()）
  - 当前训练轮次与早停计数器
- 重启训练时，检测到`models/checkpoint.pt`会自动从断点恢复


## 部署与接口测试
1. 启动API服务：`python src/main.py serve`
2. 访问`http://localhost:8000/docs`，在`/predict`接口页点击「Try it out」
3. 输入商品标题，点击「Execute」，即可查看预测结果
4. 支持通过Postman、curl等工具批量调用接口，适合集成到电商商品发布系统


## 数据集说明
- 来源：百度AI Studio商品标题分类数据集
- 数据规模：包含训练集、验证集、测试集，覆盖多个电商品类
- 数据格式：
  | label（品类） | text_a（商品标题）                          |
  |--------------|---------------------------------------------|
  | 母婴         | 好奇心钻装纸尿裤L40片9-14kg                  |
  | 蔬菜         | 基地玉米                                    |
  | 酒饮冲调     | 240ML*15养元2430六个核桃                    |
  | 玩具         | 911-267遥控车                                |
  | 乳品         | 125ML*4伊利臻浓牛奶                          |


## 扩展建议
1. 可增加品类置信度输出，帮助商家判断分类可靠性
2. 支持批量预测接口，提升大批量商品处理效率
3. 增加模型压缩（如量化、剪枝），降低部署资源占用
4. 集成到电商后台，实现商品标题输入后自动填充品类

如需补充**LICENSE文件**、**Docker部署配置**或**超参数调优指南**，可随时告知！
