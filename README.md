# 智能商品分类预测系统（基于BERT）

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
- [扩展建议](#扩展建议)


## 项目简介
本项目基于预训练BERT模型（`bert-base-chinese`）实现商品标题的自动标签分类，支持从商品标题文本中识别对应的品类标签，适用于电商平台商品上架、品类管理等场景。系统具备数据预处理、模型训练、评估、预测及API服务等完整功能，可根据实际业务需求灵活配置参数。


## 核心能力
1. **数据处理**：支持对商品标题文本进行清洗、分词、标签编码及格式转换，生成模型可直接使用的输入数据
2. **模型训练**：基于BERT构建分类模型，支持冻结/微调BERT参数、早停机制、断点续训及混合精度训练
3. **性能评估**：提供准确率、精确率、召回率、F1-score等多维度指标，结合混淆矩阵分析分类效果
4. **快速预测**：支持单条文本交互式预测及批量预测，提供HTTP API服务便于集成到业务系统
5. **灵活配置**：通过配置文件统一管理路径、超参数等，支持自定义训练策略和模型参数


## 环境搭建
### 依赖安装
```bash
# 推荐Python 3.8+
pip install -r requirements.txt
# 核心依赖包括：
# - torch>=1.10.0
# - transformers>=4.10.0
# - scikit-learn>=1.0.0
# - fastapi>=0.68.0
# - uvicorn>=0.15.0
```

### 预训练模型准备
项目默认使用`bert-base-chinese`预训练模型，已存放于`pretrained/bert-base-chinese/`目录，包含：
- 模型权重文件（`pytorch_model.bin`）
- 配置文件（`config.json`）
- 词汇表（`vocab.txt`，支持中文、英文、数字及常见符号分词）


## 快速上手
### 1. 数据预处理
```bash
python src/main.py --mode process
# 功能：处理data/raw/下的train.txt/valid.txt/test.txt
# 输出：生成data/processed/下的预处理数据（含分词后特征及标签编码）
```

### 2. 模型训练
```bash
python src/main.py --mode train --config configs/train_config.json
# 功能：基于训练集训练模型，使用验证集评估并保存最优模型
# 输出：模型文件保存至models/（含最优模型model.pt和断点checkpoint.pt）
```

### 3. 模型评估
```bash
python src/main.py --mode evaluate --model_path models/model.pt
# 功能：在测试集上评估模型性能
# 输出：打印准确率、F1-score等指标，生成混淆矩阵可视化结果
```

### 4. 交互式预测
```bash
python src/main.py --mode predict --model_path models/model.pt
# 功能：输入商品标题，返回预测的品类标签及置信度
```

### 5. 启动API服务
```bash
python src/main.py --mode serve --host 0.0.0.0 --port 8000
# 功能：启动FastAPI服务，提供HTTP预测接口
# 接口文档：访问http://localhost:8000/docs查看
```


## 项目结构
```
Classifying_merchant_product_releases_using_the_BERT_model/
├── configs/               # 配置文件（训练参数、路径等）
├── data/
│   ├── raw/               # 原始数据（train.txt/valid.txt/test.txt）
│   └── processed/         # 预处理后的数据
├── logs/                  # TensorBoard日志
├── models/                # 训练好的模型文件
├── pretrained/            # 预训练模型（bert-base-chinese）
├── src/
│   ├── configuration/     # 配置加载工具
│   ├── model/             # 模型定义（BERTClassifier）
│   ├── preprocess/        # 数据预处理逻辑
│   ├── runner/            # 训练、评估、预测流程
│   ├── web/               # FastAPI服务实现
│   └── main.py            # 项目入口
├── requirements.txt       # 依赖清单
└── README.md              # 项目说明
```


## 使用指南
### 配置文件说明
核心配置文件（`configs/train_config.json`）关键参数：
- `model_name_or_path`：预训练模型路径（默认`pretrained/bert-base-chinese`）
- `num_classes`：分类类别数量（根据数据集自动生成）
- `freeze_bert`：是否冻结BERT参数（`true`/`false`）
- `epochs`：最大训练轮次
- `batch_size`：批次大小
- `learning_rate`：学习率
- `patience`：早停机制耐心值（默认2）

### 数据格式要求
原始数据（`data/raw/`）需为`label\ttext_a`格式（制表符分隔），示例：
```
家电  智能电饭煲5L大容量
服装  纯棉短袖T恤男
```


## 模型优化方案
1. **训练策略**：
   - 采用交叉熵损失函数，结合Adam优化器
   - 支持学习率衰减和混合精度训练（`torch.amp`）
   - 早停机制避免过拟合（基于验证集F1-score）

2. **性能提升**：
   - 断点续训：支持从最近 checkpoint 恢复训练
   - 设备自适应：自动选择GPU（cuda）或CPU训练


## 部署与接口测试
### API接口说明
启动服务后提供以下接口：
- `POST /predict`：单条预测，输入`{"text": "商品标题"}`，返回`{"label": "品类", "confidence": 0.95}`

### 测试示例
```bash
# 单条预测
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "无线蓝牙耳机"}'
```


## 数据集说明
- 来源：百度AI Studio商品标题分类数据集（https://aistudio.baidu.com/datasetdetail/135131）
- 内容：包含训练集（train.txt）、验证集（valid.txt）、测试集（test.txt）
- 预处理：过滤空值、标签标准化、BERT分词（最大长度由配置文件指定）


## 扩展建议
1. 增加品类置信度阈值过滤，提升分类可靠性
2. 优化批量预测接口，支持文件上传（如CSV）批量处理
3. 模型压缩：通过量化（INT8）、剪枝减少模型体积和推理耗时
4. 集成电商后台：对接商品发布系统，实现品类自动填充
5. 多模型融合：结合文本特征与商品属性（如价格、图片）提升分类精度


如需补充**LICENSE文件**、**Docker部署配置**或**超参数调优指南**，可随时告知！


以上内容基于项目实际实现调整，重点修正了目录结构、功能描述与代码模块的对应关系，补充了接口细节和数据格式说明，如需进一步调整某个模块的描述，可以随时告诉我具体差异点~
