import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

from config import configs
from model.bert_classifier import BERTClassifier
from preprocess.data_load import get_dataloader, DatasetType


def evaluate_model(model, dataloader, device ,max_eval_batch = 30):   #max_eval_batch: 是表示在训练过程中取多少个batch进行评估一次，如果整体评估，那么就不需要这个参数
    '''
    评估模型性能
    :param model: 模型
    :param dataloader:评估数据加载器
    :param device: 设备
    :return: 返回评估结果----一个字典，包含准确率、精确率、召回率和F1分数
    '''
    # 初始化列表，收集所有样本的真实标签和预测标签（全局统计，保证macro平均准确）
    all_labels = []
    all_preds = []
    # 遍历dataloader，控制最大批次
    for batch_idx, batch in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        # 达到指定批次，停止评估
        if max_eval_batch is not None and batch_idx >= max_eval_batch:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].tolist()  # [batch_size]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1).tolist()  # [batch_size]，预测标签
            # 收集当前batch的标签（全局累计）
            all_labels.extend(labels)
            all_preds.extend(preds)

    # 计算全局指标（基于所有收集到的样本）
    result = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }

    # 打印评估信息（告知实际评估了多少样本和批次）
    print(f"实际评估批次：{min(max_eval_batch, len(dataloader)) if max_eval_batch else len(dataloader)}")
    print(f"实际评估样本数：{len(all_labels)}")

    return result


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
    result = evaluate_model(model, dataloader, device , max_eval_batch = 10)
    print(f"准确率是: {result['accuracy']:.4f}")
    print(f"精确率是: {result['precision']:.4f}")
    print(f"召回率是: {result['recall']:.4f}")
    print(f"F1分数是: {result['f1_score']:.4f}")


if __name__ == "__main__":
    run_evaluate()




