import torch
import numpy as np
import LoadData
#from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, pipeline, RobertaTokenizer
from transformers import BertModel, RobertaModel
import re
import nltk
import nltk.tokenize as tk
from punctuationset import *
#nltk.download('punkt')
from sentic_compute import load_related_word

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

wnl = WordNetLemmatizer()
lemmas_sent = []
from sentic_compute import load_sentic_word, load_sentic_7_word
senticNet = load_sentic_word()
senticNet_7 = load_sentic_7_word()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#sentence_sentiment_polarity = pipeline("sentiment-analysis")

#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-uncased')
#加载预训练模型
pretrained = BertModel.from_pretrained("bert-base-uncased").to(device)

# token = RobertaTokenizer.from_pretrained('roberta-base')
# pretrained = RobertaModel.from_pretrained("roberta-base").to(device)

#不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

class ExtractTextFeature(torch.nn.Module):
    def __init__(self,text_length,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size  # 256
        self.text_length = text_length  # 75
        self.fc = torch.nn.Linear(768, hidden_size*2)

    def word_split(self, sentence):
        word_list = tokenizer.tokenize(sentence)
        for word in word_list:
            if word in punctuation:
                word_list.remove(word)

        return word_list

    def forward(self,text):

        data = token.batch_encode_plus(batch_text_or_text_pairs=text,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=75,
                                       return_attention_mask=True,
                                       return_token_type_ids=True,
                                       return_tensors='pt',
                                       return_length=True)
        #print("--------- ", type(data))
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)

        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            # out = pretrained(input_ids=input_ids,
            #                  attention_mask=attention_mask )

        result=self.fc(out[-1])

        # out[0]：[512]
        seq = self.fc(out[0])
        # word_lit = []
        # for sent in text:
        #     sent_lit = self.word_split(sent)
        #     for
        #
        #
        #     word_lit.append(self.word_split(sent))


        return result,seq


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    if tag.startswith('V'):
        return wordnet.VERB
    if tag.startswith('N'):
        return wordnet.NOUN
    if tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


class sentiment_polarity(torch.nn.Module): #先分辨距自己，在分别词级，之后将其拼接后通过激活函数
    def __init__(self, batch_size):
        super(sentiment_polarity, self).__init__()
        self.batch_size = batch_size
       # self.sentence_sentiment_polarity = pipeline("sentiment-analysis")
        # self.mlplayer = torch.nn.Sequential(
        #     torch.nn.Linear(200,200),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.2),
        #     torch.nn.Linear(200, 200),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.2)
        # )
        # self.sentence_polarity_linear = torch.nn.Linear(1, 10, bias=False)
        self.word_level_polarity_linear = torch.nn.Linear(300, 300, bias=False)

    def word_split(self, sentence):
        word_list = tokenizer.tokenize(sentence)
        for word in word_list:
            if word in punctuation:
                word_list.remove(word)

        return word_list


    def forward(self,text): #text是32句话
        #text_polarity = self.sentence_sentiment_polarity(text)
        sentence_polarity_list = []
        word_level_polarity_list = []
        #with torch.no_grad():
        for sentence in text: #对于一句话
            #将句子分词
            word_list = self.word_split(sentence)

            # lemmas_sent = []
            # tagged_sent = pos_tag(word_list)
            # for tag in tagged_sent:
            #     # print("1111  ", type(tag),tag)
            #     wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            #     lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
            # word_list = lemmas_sent

            sentence_word_polarity = []

            for word in word_list:
                if word.lower() in senticNet_7:
                    tt = word.lower()
                    sentence_word_polarity.append(senticNet_7[str(tt)]+1)
                else:
                    if word.lower() in senticNet:
                        tt = word.lower()
                        sentence_word_polarity.append(senticNet[str(tt)] + 1)
                    else:
                        sentence_word_polarity.append(1)


            # for word in word_list:  # 这个地方改一下
            #     if word.lower() in senticNet:
            #         tt = word.lower()
            #         sentence_word_polarity.append(senticNet[str(tt)] + 1)
            #     else:
            #         sentence_word_polarity.append(1)



            if len(sentence_word_polarity) >= 300:
                sentence_word_polarity = sentence_word_polarity[:300]
            else:
                list_pad = [0] * (300 - len(sentence_word_polarity))
                sentence_word_polarity.extend(list_pad)

            word_level_polarity_list.append(np.array(sentence_word_polarity))

        # sentence_polarity_list = np.array(sentence_polarity_list)
        # sentence_polarity_list = torch.from_numpy(sentence_polarity_list).to(device)
        # sentence_polarity_list = torch.unsqueeze(sentence_polarity_list, dim=1).to(device).float() #->(batchsize,1)
        word_level_polarity_list = np.array(word_level_polarity_list)
        word_level_polarity_list = torch.from_numpy(word_level_polarity_list).to(device).float() #->(batchsize,length)
        #sentence_polarity_list = self.sentence_polarity_linear(sentence_polarity_list)
        word_level_polarity_list = self.word_level_polarity_linear(word_level_polarity_list)
        # text_polarity_raw = torch.cat((sentence_polarity_list, word_level_polarity_list), dim=1) #(batchsize, 200) 句子的和词的平起来，不经过全连接
        # text_polarity = self.mlplayer( text_polarity_raw ) #100 通过全联接
        #return text_polarity, text_polarity_raw
        return word_level_polarity_list, 0


if __name__ == "__main__":
    # model = ExtractTextFeature(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN).to(device)
    # for text, text_index,image_feature,attribute_index,group,id in LoadData.train_loader:
    #
    #     out,seq = model(text)
    #     print(out.shape) #[32, 512]
    #     print(seq.shape) #[32, 75, 512]
    #
    #     break

    p = sentiment_polarity(1)
    list = p("we")
    print("p = sentiment_polarity() ", list[0]['label'], list[0]['score'])
