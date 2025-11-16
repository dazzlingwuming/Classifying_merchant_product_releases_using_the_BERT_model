import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import configs
from model.bert_classifier import BERTClassifier
from preprocess.data_load import get_dataloader, DatasetType
from runner.evaluate import evaluate_model


def one_epoch_train(model, train_dataloader , loss_fn, optimizer, device, epoch, writer,best_loss ,scaler):
    total_loss = 0
    for step, batch in tqdm(enumerate(train_dataloader), desc=f"Training Epoch {epoch + 1}", total=len(train_dataloader)):
        model.train()
        inputs = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 梯度清零
        optimizer.zero_grad()

        #使用混合精度训练
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            # 前向传播
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            # 计算损失
            loss = loss_fn(outputs, labels)
        # # 反向传播
        # loss.backward()
        # # 参数更新
        # optimizer.step()
        # 使用混合精度的反向传播和优化步骤
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if (step + 1) % 30 == 0:
            avg_loss = total_loss / 30
            print(f"Epoch [{epoch + 1}/{configs.config_train['num_epochs']}], Step [{step + 1}/{len(train_dataloader)}], Loss: {avg_loss:.4f}")
            writer.add_scalar('Training Loss', avg_loss, epoch * len(train_dataloader) + step)
            total_loss = 0
            #保存模型,这里是保存的训练过程中表现最好的模型，但是可能会有过拟合风险，所以可以考虑保存验证集上表现最好的模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), configs.Best_Model_PATH)
                print(f"Model saved at epoch {epoch + 1} with train_loss {best_loss:.4f}")


def one_epoch_eval(model, eval_dataload, device, writer ,epoch,max_eval_batch = 30):
    model.eval()
    with torch.no_grad():
        results = evaluate_model(model, eval_dataload, device, max_eval_batch=max_eval_batch)
        print(f"Evaluation Results - Accuracy: {results['accuracy']:.4f}, Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F1 Score: {results['f1_score']:.4f}")
        writer.add_scalar('Eval Accuracy', results['accuracy'], epoch)
        writer.add_scalar('Eval Precision', results['precision'], epoch )
        writer.add_scalar('Eval Recall', results['recall'], epoch )
        writer.add_scalar('Eval F1 Score', results['f1_score'], epoch)
        f1 = results["f1_score"]
    return f1


def train():

    #设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #数据加载
    train_dataload = get_dataloader(data_type= DatasetType.Train)
    eval_dataload = get_dataloader(data_type= DatasetType.Validation)

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

    #混合精度初始化
    use_amp = True
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    #评估容忍度
    patience = 2
    patience_counter = 0

    #训练循环
    for epoch in range(configs.config_train["num_epochs"]):
        best_f1 = 0
        model.train()
        best_loss = float('inf')
        one_epoch_train(model, train_dataload,loss_fn, optimizer, device, epoch, writer,best_loss ,scaler)
        f1 = one_epoch_eval(model, eval_dataload, device, writer ,epoch , max_eval_batch = 30)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), configs.Best_Model_EVAL_PATH)
            print(f"Model saved at epoch {epoch + 1} with best_f1 {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    writer.close()



if __name__ == "__main__":

    pass