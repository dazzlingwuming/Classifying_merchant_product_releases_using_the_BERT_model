import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

from config import configs
from model.bert_classifier import BERTClassifier
from preprocess.data_load import get_dataloader, DatasetType


def evaluate_model(model, dataloader, device):
    '''
    评估模型性能
    :param model: 模型
    :param dataloader:评估数据加载器
    :param device: 设备
    :return: 返回评估结果----一个字典，包含准确率、精确率、召回率和F1分数
    '''
    for batch in tqdm(dataloader , desc="Evaluating"):
        accuracy = 0 #准确率
        precision = 0 #精确率
        recall = 0 #召回率
        f1 = 0 #F1分数
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].tolist() #[batch_size]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = torch.argmax(outputs, dim=1) #[batch_size]
            outputs = outputs.tolist()
        if accuracy == 0:
            accuracy = accuracy_score(labels, outputs)
        else:
            accuracy = (accuracy_score(labels, outputs)+ accuracy)/2
        #计算精确率
        if precision == 0:
            precision = precision_score(labels, outputs, average='macro', zero_division=0)
        else:
            precision = (precision_score(labels, outputs, average='macro', zero_division=0) + precision)/2
        #计算召回率
        if recall == 0:
            recall = recall_score(labels, outputs, average='macro' , zero_division=0)
        else:
            recall = (recall_score(labels, outputs, average='macro' , zero_division=0) + recall)/2
        #计算F1分数
        if f1 == 0:
            f1 = f1_score(labels, outputs, average='macro', zero_division=0)
        else:
            f1 = (f1_score(labels, outputs, average='macro', zero_division=0) + f1)/2
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def run_evaluate():
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = BERTClassifier()
    model.load_state_dict(torch.load(configs.Best_Model_PATH, map_location=device))
    # 设置模型为评估模式
    model.to(device)
    model.eval()

    #加载测试数据
    dataloader = get_dataloader(data_type= DatasetType.Test)

    #评估模型
    result = evaluate_model(model, dataloader, device)
    print(f"准确率是: {result['accuracy']:.4f}")
    print(f"精确率是: {result['precision']:.4f}")
    print(f"召回率是: {result['recall']:.4f}")
    print(f"F1分数是: {result['f1_score']:.4f}")




