from datasets import load_dataset, ClassLabel
import pandas as pd
from transformers import AutoTokenizer

from config import configs

def tokenize_split(examples,tokenizer):
    '''
    :param examples:
    :param tokenizer:
    :return:
    '''
    tokenizers = tokenizer(examples["text_a"], padding=configs.MAX_SEQ_LENGTH_SECTIONS, truncation=True , max_length=configs.MAX_SEQ_LENGTH)
    '''
    padding 
    True/'longest'	仅 padding 到当前 batch 中最长序列的长度（默认行为）
    'max_length'	强制 padding 到 max_length 指定的固定长度
    False	        不 padding
    '''
    return {
        "input_ids": tokenizers["input_ids"],
        "attention_mask": tokenizers["attention_mask"]
    }

def load_data_processor():
    data_dic = load_dataset(
        "csv",
        data_files={
            "train": str(configs.RAW_DATA_DIR/ "train.txt"),
            "validation": str(configs.RAW_DATA_DIR/ "train.txt"),
            "test": str(configs.RAW_DATA_DIR/ "train.txt")
        },
        delimiter="\t"
    )#data_dic是一个DatasetDict对象，包含train、validation、test三个数据集,并且每一个数据集都是一个二元组，分别包含text_a和label两个字段

    #对数据集进行预处理，将text_a字段中的文本进行清洗，去除多余的空格和特殊字符
    data_dic = data_dic.filter(lambda y : y['text_a'] is not None and y['label'] is not None)

    #加载字典
    tokenizer = AutoTokenizer.from_pretrained(configs.PRETRAINED_MODELS_DIR/"bert-base-chinese")

    #对数据集进行分词和编码
    data_dic = data_dic.map(lambda x: tokenize_split(x,tokenizer), batched=True , remove_columns=["text_a"])
    #用于对数据集中的每个样本或批量样本应用一个函数，并返回一个新的数据集。

    #对于标签值进行类型转换，将其转换为整数类型，利用datasets库的里面的ClassLabel特性
    #先统计标签的种类
    # unique_labels = set(data_dic['train']['label'])
    # #写入文档中
    # with open(configs.Label_Voc,"w",encoding='utf-8') as write_label:
    #     for label in unique_labels:
    #         write_label.writelines(str(label)+'\n')

    #读取构建好的标签词典
    label_list = []
    with open(configs.Label_Voc,"r",encoding='utf-8') as read_label:
        for line in read_label:
            label_list.append(line.strip())

    #将标签映射为整数
    class_label = ClassLabel(names=label_list)
    data_dic = data_dic.cast_column("label", class_label)
    '''
    将 data_dic 数据集中的 label 列转换为 ClassLabel 类型。
    ClassLabel 是 datasets 提供的一种特殊类型，用于表示分类标签，它可以将字符串标签映射为整数，或者将整数映射回字符串标签。
    '''
    # print(data_dic[:3])

    # data_dic["train"].features
    '''
    data_dic["train"].features 返回的是 datasets 库中 Dataset 对象的特征信息。它是一个 Features 对象，描述了数据集中每一列的名称和类型。
    其实就是一个字典，键是列名，值是对应的特征类型（如 ClassLabel、Value、Sequence 等）。
    于是在想查询类别标签的具体信息时，可以通过 data_dic["train"].features["label"] 来访问标签列的特征信息。
    '''

    #保存预处理后的数据集到本地
    data_dic["train"].save_to_disk(str(configs.PROCESSED_DIR/"train"))
    data_dic["validation"].save_to_disk(str(configs.PROCESSED_DIR/"validation"))
    data_dic["test"].save_to_disk(str(configs.PROCESSED_DIR/"test"))
    '''
    save_to_disk将数据集保存到本地磁盘，生成三个文件，一个arrow文件，一个json文件，一个state文件。
    arrow文件存储数据集的实际数据，json文件存储数据集的元数据（如列名、类型等），state文件存储数据集的状态信息（如版本号等）。
    '''



    pass


if __name__ == "__main__":
    load_data_processor()