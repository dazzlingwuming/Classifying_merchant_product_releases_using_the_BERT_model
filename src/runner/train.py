import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import configs
from model.bert_classifier import BERTClassifier
from preprocess.data_load import get_dataloader, DatasetType


def train():

    #设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #数据加载
    dataload = get_dataloader(data_type= DatasetType.Train)

    #模型加载
    model = BERTClassifier()

    #损失函数和优化器定义
    loss_fn  = torch.nn.CrossEntropyLoss()

    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.config_train["learning_rate"],)

    #log设置
    writer = SummaryWriter(log_dir= configs.LOGS_DIR/f"train_{time.strftime('%Y%m%d-%H%M%S')}")

    #设备放置
    model.to(device)

    #训练循环
    for epoch in range(configs.config_train["num_epochs"]):
        model.train()
        total_loss = 0
        best_loss = float('inf')
        avg_loss = 0
        for step , batch in tqdm(enumerate(dataload) , desc= "Training Epoch {}".format(epoch+1), total=len(dataload)):
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            #梯度清零
            optimizer.zero_grad()
            #前向传播
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            #计算损失
            loss = loss_fn(outputs, labels)
            #反向传播
            loss.backward()
            #参数更新
            optimizer.step()
            total_loss += loss.item()
            if (step + 1) % 30 == 0:
                avg_loss = total_loss / 10
                print(f"Epoch [{epoch+1}/{configs.config_train['num_epochs']}], Step [{step+1}/{len(dataload)}], Loss: {avg_loss:.4f}")
                writer.add_scalar('Training Loss', avg_loss, epoch * len(dataload) + step)
                total_loss = 0
                # 保存模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), configs.Best_Model_PATH)
                    print(f"Model saved at epoch {epoch + 1} with loss {best_loss:.4f}")

    writer.close()



if __name__ == "__main__":

    pass