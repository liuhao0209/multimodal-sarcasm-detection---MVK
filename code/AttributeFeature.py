import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, pipeline
from transformers import BertModel

import LoadData


from transformers import pipeline
import nltk
import nltk.tokenize as tk
from punctuationset import *
from sentic_compute import load_sentic_word, load_sentic_7_word
senticNet = load_sentic_word()
senticNet_7 = load_sentic_7_word()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-uncased')
#加载预训练模型
pretrained = BertModel.from_pretrained("bert-base-uncased").to(device)

# token = RobertaTokenizer.from_pretrained('roberta-base')
# pretrained = RobertaModel.from_pretrained("roberta-base").to(device)

for param in pretrained.parameters():
    param.requires_grad_(False)


class ExtractAttributeFeature(torch.nn.Module):
    def __init__(self):
        super(ExtractAttributeFeature, self).__init__()
        embedding_weight=self.getEmbedding()  #一个二维张量，每行上的参数
        self.embedding_size=embedding_weight.shape[1]   #每行上的参数数量
        self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)  #在NLP任务中，当我们搭建网络时，第一层往往是嵌入层，对于嵌入层有两种方式初始化embedding向量，
        # 一种是直接随机初始化，另一种是使用预训练好的词向量初始化，
        # embedding size 200->100
        """Raw attribute vectors e(ai) are passed through a two-layer neural network to obtain the attention weights αi for constructing the attribute guidance vector vattr."""
        self.bert_embedding_size = 768
        self.Linear_1 = torch.nn.Linear(self.embedding_size, int(self.embedding_size/2))
        # embedding size 100->1
        self.Linear_2 = torch.nn.Linear(int(self.embedding_size/2),1)
        self.ac_relu = torch.nn.LeakyReLU(0.05)
        self.bertExtract_dim_out = 512
        self.Linear_bert_0 = torch.nn.Linear(self.bert_embedding_size, int(self.bertExtract_dim_out))
        self.Linear_bert_1 = torch.nn.Linear(self.bert_embedding_size, int(self.bertExtract_dim_out))



        self.hidden_size = 100  # 256
        self.text_length = 100  # 75
        self.fc = torch.nn.Linear(768, self.hidden_size * 2)

    def forward(self, input):
        """
        e(a_i)
        """
        # -1 represent batch size ,在torch里面，view函数相当于numpy的reshape，  (batch size,5,embedding_size)

        """
        self.embedded = self.embedding(input).view(-1, 5, self.embedding_size) #torch.Size([256, 5, 200])
        
        print("-------- ", input.size(), self.embedded.size())
        
        data = token.batch_encode_plus(batch_text_or_text_pairs=input,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=5,
                                       return_attention_mask=True,
                                       return_token_type_ids=True,
                                       return_tensors='pt',
                                       return_length=True)
        #print("--------- ", type(data))
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        """
        with torch.no_grad():
            self.bert_embedded = pretrained(input_ids=input,
                             attention_mask=None,
                             token_type_ids=None)
            
             # self.bert_embedded = pretrained(input_ids=input_ids,
            #                  attention_mask=attention_mask )
        self.embedded = self.ac_relu(self.Linear_bert_0(self.bert_embedded[0]))

        #attn_weights = self.Linear_1(self.embedded.view(-1,self.embedding_size))
        # attn_weights = torch.nn.functional.tanh(attn_weights)
        #attn_weights = self.ac_relu(attn_weights)
        #attn_weights = self.Linear_2(attn_weights)
        """
        a=softmax(a) 对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1
        dim:指明维度，dim=0表示按列计算；dim=1表示按行计算。默认dim的方法已经弃用了，最好声明dim，否则会警告
        """
        #attn_weights = torch.nn.functional.softmax(attn_weights.view(-1,5),dim=1)
        #finalState = torch.bmm(attn_weights.unsqueeze(1), self.embedded).view(-1,200)   #计算两个tensor的矩阵乘法，torch.bmm(a,b),
        finalState = self.ac_relu(self.Linear_bert_1(self.bert_embedded[1]))
        return finalState, self.embedded

    def getEmbedding(self):   #torch. from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
        # numpy.loadtxt() 函数从文本文件中加载数据,返回值为从 txt 文件中读取的 N 维数组。
        return torch.from_numpy(np.loadtxt("multilabel_database_embedding/vector.txt", delimiter=' ', dtype='float32'))


def word_convert(text):
    text_temp = []
    bat = len(text[0])
    for i in range(bat):
        wordlist = []
        for j in range(5):
            wordlist.append(text[j][i])
        text_temp.append(wordlist)
    #text = text_temp
    text = [' '.join(word) for word in text_temp]
    return text


class attribute_sentiment_polarity(torch.nn.Module):
    def __init__(self, batchsize):
        super(attribute_sentiment_polarity, self).__init__()
        self.batchsize = batchsize
        #self.attribution_sentiment_polarity = pipeline("sentiment-analysis")
        self.mlplayer = torch.nn.Sequential(
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

    def forward(self, text):
        text_temp = []
        bat = len(text[0])
        for i in range(bat):
            wordlist = []
            for j in range(5):
                wordlist.append( text[j][i] )
            text_temp.append(wordlist)
        text = text_temp #转为[32, 5]
        text_polarity_list = []

        for attr_word_list in text:
            attr_word_polarity_list = []
            for attr_word in attr_word_list:
                if attr_word.lower() in senticNet_7:
                    tt = attr_word.lower()
                    attr_word_polarity_list.append(senticNet_7[str(tt)]+1)
                else:
                    if attr_word.lower() in senticNet:
                        tt = attr_word.lower()
                        attr_word_polarity_list.append(senticNet[str(tt)] + 1)
                    else:
                        attr_word_polarity_list.append(1)


                # if attr_word.lower() in senticNet:
                #     tt = attr_word.lower()
                #     attr_word_polarity_list.append(senticNet[str(tt)]+1)
                # else:
                #     attr_word_polarity_list.append(1)

            text_polarity_list.append(np.array(attr_word_polarity_list))

        text_polarity_list = np.array(text_polarity_list)
        text_polarity_list = torch.from_numpy(text_polarity_list).to(device).float()
        #print("text_polarity_list ", text_polarity_list.size())
        attr_polarity = self.mlplayer(text_polarity_list)

        #return attr_polarity #(batchsize, 5)
        return text_polarity_list


if __name__ == "__main__":
    test=ExtractAttributeFeature()
    for text,text_index,image_feature,attribute_index,group,id in LoadData.train_loader:
        print(LoadData.train_loader.shape)
        #attribute_index [32, 5]
        result,seq=test(attribute_index)
        # [32, 200]
        print(result.shape)
        # [32, 5, 200]
        print(seq.shape)
        break


