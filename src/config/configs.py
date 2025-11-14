# 导入路径处理所需的库
from pathlib import Path

# 定义项目根目录：当前脚本文件的上三级目录（假设此脚本位于项目子文件夹中，如 src/utils/ 下）
ROOT_DIR = Path(__file__).parent.parent.parent

# 定义数据相关目录
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw_data'       # 原始数据目录：存放未经处理的初始数据（如 CSV、Excel、JSON 等）
PROCESSED_DIR = ROOT_DIR / 'data' / 'process_data' # 处理后数据目录：存放清洗、转换后的可用数据
MAX_SEQ_LENGTH_SECTIONS = 'max_length'                  # 是否按段落划分最大序列长度：用于文本数据处理时的段落划分设置
MAX_SEQ_LENGTH = 64                             # 最大序列长度：用于文本数据处理时的最大长度设置
Label_Voc =  RAW_DATA_DIR/'label_voc.txt'           # 标签词汇表文件路径：存放标签与其对应索引的映射关系
BATCH_SIZE = 32                                   # 批处理大小：用于模型训练和评估时的批次大小设置

#定义预训练模型目录
PRETRAINED_MODELS_DIR = ROOT_DIR / 'pretrained'  # 预训练模型目录：存放预训练模型文件（如 BERT、GPT 等）

# 定义日志与模型目录
LOGS_DIR = ROOT_DIR / 'logs'                    # 日志目录：存放项目运行日志（如训练日志、错误日志）
MODELS_DIR = ROOT_DIR / 'models'                # 模型目录：存放训练完成的模型文件（如 .pkl、.h5、.pt 等）
OUTPUTS_DIR = ROOT_DIR / 'outputs'              # 输出目录：存放项目生成的结果文件（如报告、可视化图表）
Best_Model_PATH = OUTPUTS_DIR / 'best_model.pth'    # 最佳模型路径：存放训练过程中表现最好的模型文件
CONFIG_DIR = ROOT_DIR / 'config'                # 配置目录：存放项目配置文件（如 .yaml、.json 格式的参数配置）

#模型相关内部参数
config_model = {
            "Num_classes" : 30 ,# 分类类别数：用于分类任务中不同类别的数量
            "freeze_bert" :True ,  # 是否冻结BERT模型参数：在微调过程中是否保持预训练模型参数不变
                }

#训练相关参数
config_train = {
            "num_epochs" : 5 ,      # 训练轮数：模型训练的总迭代次数
            "learning_rate" : 1e-5, # 学习率：优化器的学习速率
            "weight_decay" : 0.01,  # 权重衰减：用于正则化以防止过拟合
                }

# 定义输出与配置目录（可选，根据项目需求补充）


