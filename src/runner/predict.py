from flask import Flask,request , jsonify
import torch
from transformers import AutoTokenizer

from config import configs
from model.bert_classifier import BERTClassifier

app = Flask(__name__)
#模型和一些参数先全局加载好，避免每次调用预测函数都加载模型

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载模型
model = BERTClassifier()
model.load_state_dict(torch.load(configs.Best_Model_PATH, map_location=device))
#设置模型为评估模式
model.to(device)
model.eval()

#加载分词器
tokenizer = AutoTokenizer.from_pretrained(configs.PRETRAINED_MODELS_DIR/"bert-base-chinese")

@app.route("/predict", methods=["POST",'GET'])
def run_predict():
    '''
    Flask接口预测函数
    '''
    if request.method == "POST":
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        predicted_label = predict(text)
        return jsonify({"predicted_label": predicted_label})
    elif request.method == "GET":
        text = request.args.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        predicted_label = predict(text)
        return jsonify({"predicted_label": predicted_label})
    else:
        return "Please use POST or GET method to submit data."


def predict(text):
    '''
    预测函数
    :param 商品描述 text: str
    :return: 预测结果
    '''
    #对输入文本进行分词和编码
    inputs = tokenizer(text, return_tensors="pt", padding=configs.MAX_SEQ_LENGTH_SECTIONS, truncation=True, max_length=configs.MAX_SEQ_LENGTH)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    #前向传播
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #获取预测结果
        predicted_class = torch.argmax(outputs, dim=1).item()

    #找到对应的标签
    label_list = []
    with open(configs.Label_Voc,"r",encoding='utf-8') as read_label:
        for line in read_label:
            label_list.append(line.strip())
    predicted_label = label_list[predicted_class]
    return predicted_label


@app.route("/", methods=["POST",'GET'])
def home():
    return "Welcome to the Text Classification API!"


if __name__ == "__main__":
    # A = predict("手机是一种便携式电子设备，具有通信、娱乐和计算等多种功能。")
    # print(A)
    # 创建Flask应用

    # 运行Flask应用
    app.run(host="0.0.0.0", port=5000)









