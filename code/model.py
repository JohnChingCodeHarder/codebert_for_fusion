# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)

        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob

    def get_feature_vector(self, input_ids=None):
        # 获取RobertaModel的输出
        outputs = self.encoder.roberta(
            input_ids, 
            attention_mask=input_ids.ne(1),
            output_hidden_states=True  # 启用输出所有隐藏状态
        )
        
        # 获取最后一层的隐藏状态
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        print("原始输出维度:", last_hidden_state.shape)
        
        # 获取[CLS]标记对应的隐藏状态
        cls_hidden_state = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        print("CLS token维度:", cls_hidden_state.shape)
        
        return cls_hidden_state  # 返回768维特征向量



