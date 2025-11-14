from torch import nn
from transformers import BertModel

from config import configs #其实按道理可以将configs设置成一个类，直接导入类实例，不过目前这样也行

#
class BERTClassifier(nn.Module):
    def __init__(self):
        super(BERTClassifier, self).__init__()
        #首先定义BERT模型
        self.bert = BertModel.from_pretrained(configs.PRETRAINED_MODELS_DIR/'bert-base-chinese')
        #是否冻结BERT模型参数
        if configs.config_model["freeze_bert"]:
            for param in self.bert.parameters():
                param.requires_grad = False
        #定义分类器的线性层
        self.linear = nn.Linear(self.bert.config.hidden_size, configs.config_model["Num_classes"])



    def forward(self, input_ids,attention_mask):
        '''
        前向传播函数
        :param input_ids:  [batch_size, seq_length , hidden_size]
        :param attention_mask:[batch_size, seq_length , hidden_size]
        :return: [batch_size, num_classes]
        '''
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)#output [batch_size, seq_length , hidden_size]
        last_hidden_states = outputs.last_hidden_state  # 获取最后一层隐藏状态 [batch_size, seq_length , hidden_size]
        #取[CLS]标记对应的隐藏状态作为句子的表示
        last_hidden_states = last_hidden_states[:, 0, :]  # [batch_size,hidden_size]
        logits = self.linear(last_hidden_states)
        return logits
