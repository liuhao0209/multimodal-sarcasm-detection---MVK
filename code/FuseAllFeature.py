import torch
import math
import LoadData
import TextFeature
import AttributeFeature
import ImageFeature
import FinalClassifier
from LoadData import *
import torch.nn as nn
from TransEncoder import TransformerEncoder
import torch.nn.functional as F
from AttributeFeature import attribute_sentiment_polarity
from CrossAttention import *
from ViT_Transfomer import ViTTransformer, PositionWiseFeedForward, PositionWiseFeedForward_change_dim, TowertransformerEncoder
from MUSEAttention import MUSEAttention
from ExternalAttention import ExternalAttention
from SelfAttention import ScaledDotProductAttention



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#只针对一个特征
class RepresentationFusion(torch.nn.Module):
    def __init__(self,att1_feature_size,att2_feature_size,att3_feature_size):
        super(RepresentationFusion, self).__init__()
        self.linear1_1 = torch.nn.Linear(att1_feature_size+att1_feature_size, int((att1_feature_size+att1_feature_size)/2))
        self.linear1_2 = torch.nn.Linear(att1_feature_size+att2_feature_size, int((att1_feature_size+att2_feature_size)/2))
        self.linear1_3 = torch.nn.Linear(att1_feature_size+att3_feature_size, int((att1_feature_size+att3_feature_size)/2))
        self.linear2_1 = torch.nn.Linear(int((att1_feature_size+att1_feature_size)/2), 1)
        self.linear2_2 = torch.nn.Linear(int((att1_feature_size+att2_feature_size)/2), 1)
        self.linear2_3 = torch.nn.Linear(int((att1_feature_size+att3_feature_size)/2), 1)

    def forward(self, feature1,feature2,feature3,feature1_seq):
        output_list_1=list()
        output_list_2=list()
        output_list_3=list()
        length=feature1_seq.size(0)
        for i in range(length):
            output1=torch.tanh(self.linear1_1(torch.cat([feature1_seq[i],feature1],dim=1)))
            output2=torch.tanh(self.linear1_2(torch.cat([feature1_seq[i],feature2],dim=1)))
            output3=torch.tanh(self.linear1_3(torch.cat([feature1_seq[i],feature3],dim=1)))
            output_list_1.append(self.linear2_1(output1))
            output_list_2.append(self.linear2_2(output2))
            output_list_3.append(self.linear2_3(output3))
        weight_1=torch.nn.functional.softmax(torch.torch.stack(output_list_1),dim=0)
        weight_2=torch.nn.functional.softmax(torch.torch.stack(output_list_2),dim=0)
        weight_3=torch.nn.functional.softmax(torch.torch.stack(output_list_3),dim=0)
        output=torch.mean((weight_1+weight_2+weight_3)*feature1_seq/3,0)
        return output

class ModalityFusion(torch.nn.Module):
    def __init__(self,image_dim, text_dim, attribute_dim):
        super(ModalityFusion, self).__init__()
        image_feature_size=image_dim #image_feature.size(1)
        text_feature_size=text_dim #text_feature.size(1)
        attribute_feature_size=attribute_dim #attribute_feature.size(1)

        self.image_attention=RepresentationFusion(image_feature_size,text_feature_size,attribute_feature_size)
        self.text_attention=RepresentationFusion(text_feature_size,image_feature_size,attribute_feature_size)
        self.attribute_attention=RepresentationFusion(attribute_feature_size,image_feature_size,text_feature_size)

        self.image_linear_1=torch.nn.Linear(image_feature_size,250)
        self.text_linear_1=torch.nn.Linear(text_feature_size,512)
        self.attribute_linear_1=torch.nn.Linear(attribute_feature_size,512)

        self.image_linear_2=torch.nn.Linear(250,1)
        self.text_linear_2=torch.nn.Linear(512,1)
        self.attribute_linear_2=torch.nn.Linear(512,1)

        self.image_linear_3=torch.nn.Linear(image_feature_size,512)
        self.text_linear_3=torch.nn.Linear(text_feature_size,512)
        self.attribute_linear_3=torch.nn.Linear(attribute_feature_size,512)

    def forward(self, image_feature,image_seq,text_feature,text_seq,attribute_feature,attribute_seq):
        #print(" ModalityFusionfusion = self.fuse ", image_feature.size(), image_seq.size(), text_feature.size(), text_seq.size(),
        #      attribute_feature.size(), attribute_seq.size())
                                             # [32, 500]     [32, 512]      [32, 200]         [196, 32, 500]每张图片的196块全部区域
        image_vector    =self.image_attention(image_feature,text_feature,attribute_feature,image_seq)
                                             # [32, 512]     [32, 500]      [32, 200]       [75, 32, 512]
        text_vector     =self.text_attention(text_feature,image_feature,attribute_feature,text_seq)
                                                     #[32, 200]      [32, 500]     [32, 512]       [5, 32, 200]
        attribute_vector=self.attribute_attention(attribute_feature,image_feature,text_feature,attribute_seq)

        image_hidden=torch.tanh(self.image_linear_1(image_vector))
        text_hidden=torch.tanh(self.text_linear_1(text_vector))
        attribute_hidden=torch.tanh(self.attribute_linear_1(attribute_vector))

        image_score=self.image_linear_2(image_hidden)
        text_score=self.text_linear_2(text_hidden)
        attribute_score=self.attribute_linear_2(attribute_hidden)
        score=torch.nn.functional.softmax(torch.stack([image_score,text_score,attribute_score]),dim=0)

        image_vector=torch.tanh(self.image_linear_3(image_vector))
        text_vector=torch.tanh(self.text_linear_3(text_vector))
        attribute_vector=torch.tanh(self.attribute_linear_3(attribute_vector))
        # final fuse
        output=score[0]*image_vector+score[1]*text_vector+score[2]*attribute_vector
        return output

def polarity_gap(text_polarity, attribution_polarity): #attribution_polarity 5  text_polarity 200   batch_size 32 计算两种属性极性的差值
    # [32, 200] [32, 5]
    batch_size = text_polarity.size(0)
    #gap_matrix = torch.zeros([batch_size, 1000], dtype=torch.float32)
    attribute_matrix = torch.zeros([batch_size, 1000], dtype=torch.float32).to(device)
    # n = text_polarity.size(-1) / attribution_polarity.size(-1)
    # attribute_matrix = attribution_polarity.repeat(1, n)
    gap_matrix = text_polarity.repeat(1, attribution_polarity.size(1)).to(device)
    for i in range(batch_size):
        # text_polarity_g = text_polarity[i] #第 i 个数据的文本情感极性
        # g_text = text_polarity_g
        # for j in range(attribution_polarity.size(1)-1):
        #     g_text = torch.cat(g_text,text_polarity_g)
        # gap_matrix[i] = g_text
        for j in range(attribution_polarity.size(1)):
            for t in range(text_polarity.size(1)):
                attribute_matrix[i][j* text_polarity.size(1)+ t] = attribution_polarity[i][j]
    gap = torch.abs( gap_matrix - attribute_matrix )
    return gap

class middlefuseblock(nn.Module):
    def __init__(self, batchsize,fc_dropout_rate,base_model = 'LSTM', base_model_layernum = 2, embedding_dim = 100,hidden_dim=100,output_dim=200,dim_feedforword=512,num_head=4,num_layers=2,dropout = 0.2,max_len=128):
        super(middlefuseblock, self).__init__()
        #self.block = block
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dim_feedforword = dim_feedforword
        self.num_head = num_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
        self.fc_dropout_rate = fc_dropout_rate
        self.base_model = base_model
        self.batchsize = batchsize
        self.image_linear = nn.Linear(5,500)
        self.image_seq_linear = nn.Linear(130, 500)

        self.attr_linear = nn.Linear(1005,1000)

        self.TransformerEncoder_image = TransformerEncoder(embedding_dim=500, hidden_dim=500, output_dim=500, dim_feedforword=512, num_head=4, num_layers=2, dropout=0.2, max_len=1024)
        #self.TransformerEncoder_image = ViT #这个要重新写，写成和视觉相关的transformer
        self.TransformerEncoder_text = TransformerEncoder(embedding_dim=512, hidden_dim=512, output_dim=512, dim_feedforword=512, num_head=4, num_layers=2, dropout=0.2, max_len=1024)
        self.TransformerEncoder_attribution = TransformerEncoder(embedding_dim=200, hidden_dim=200, output_dim=200, dim_feedforword=512, num_head=4, num_layers=2, dropout=0.2, max_len=1024)
        self.ViTTransformerEncoder_image = ViTTransformer(num_layers=6, dims=500, num_heads=8, ff_dim=512, dropout=0.2)
        self.attention_image = nn.MultiheadAttention(embed_dim=500, num_heads=1) #这个要重新写，写成和视觉相关的注意力机制，这个用win-attention block
        self.attention_text = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
        self.attention_attribution = nn.MultiheadAttention(embed_dim=200, num_heads=1)
        #self.attention_image_seq = MUSEAttention(d_model=500, d_k=512, d_v=512, h=8)
        self.attention_image_seq = ExternalAttention(d_model=500,S=512)
        #attn_output, attn_output_weights = multihead_attn(query, key, value)

        if self.base_model == 'LSTM':
            self.base_text = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim[0], num_layers=base_model_layernum, bidirectional=True, dropout=dropout)
            self.base_attribution = nn.LSTM(input_size=200, hidden_size=hidden_dim[1], num_layers=base_model_layernum, bidirectional=True, dropout=dropout)
            #self.base_attribution = nn.Linear(200, base_model_layernum * hidden_dim[1])
            self.base_image_seq = nn.Conv1d(196, 196, 3, stride=4)
        elif self.base_model == 'GRU':
            self.base_text = nn.GRU(input_size=embedding_dim[0], hidden_size=hidden_dim[0], num_layers=base_model_layernum, bidirectional=True, dropout=dropout)
            self.base_attribution = nn.GRU(input_size=200, hidden_size=hidden_dim[1], num_layers=base_model_layernum, bidirectional=True, dropout=dropout)
            #self.base_attribution = nn.Linear(200, base_model_layernum * hidden_dim[1])
            self.base_image_seq = nn.Conv1d(196, 196, 3, stride=4)
        elif self.base_model == 'Linear':
            self.base_text = nn.Linear(embedding_dim, base_model_layernum * hidden_dim[0])
            self.base_attribution = nn.Linear(200, base_model_layernum * hidden_dim[1])
            self.base_image_seq = nn.Conv1d(196, 196, 3, stride=4)
        else:
            print('Base model must be one of LSTM/GRU/Linear')
            raise NotImplementedError

        self.text_sentiment_polarity = TextFeature.sentiment_polarity(batch_size = batchsize)
        self._attribute_sentiment_polarity = attribute_sentiment_polarity(batchsize = batchsize)

        self.Coattention = CoAttentionlayer(embedding_dim=embedding_dim)


        #------------------------------------------------------------------------------------
        self.sharetransformer = TransformerEncoder(embedding_dim=512, hidden_dim=512, output_dim=512, dim_feedforword=512, num_head=4, num_layers=2, dropout=0.2, max_len=1024)
        self.text_linear = nn.Linear(512,512)
        self.attribute_linear = nn.Linear(200,512)
        self.images_linear = nn.Linear(500,512)
        self.image_attr_linear = nn.Linear(1024+10,512+200)

        self.crossattention = nn.MultiheadAttention(embed_dim=embedding_dim+200, num_heads=4)

        self.FFW1 = PositionWiseFeedForward_change_dim(embedding_dim+200,256,256)
        #self.FFW2 = PositionWiseFeedForward(embedding_dim+200+embedding_dim+200+1000, 512)
        self.FFW2 = PositionWiseFeedForward_change_dim(256, 256, 256)

        self.selfattention = nn.MultiheadAttention(embed_dim=256, num_heads=4)

        self.text_back_linear = nn.Linear(256, 512)
        self.image_back = nn.Linear(embedding_dim+5, 500)
        self.attribute_back = nn.Linear(embedding_dim+5, 200)

        self.wt = nn.Parameter(torch.Tensor(1))
        self.wt_relu = nn.ReLU()

    def forward(self, image_result, image_seq, attribute_result, attribute_seq, attribute_words, text, text_result):
        #------------------------------------- text ---------------------------------------
        # text_trans = self.TransformerEncoder_text( torch.unsqueeze(text_result, dim=1) ) #二维转三维  [32, 1,512]
        # #print("text_trans ",text_trans.size())
        # text_base,_ = self.base_text(torch.unsqueeze(text_result, dim=1)) #[32, 1, 512]
        # #text_trans = torch.squeeze(text_trans)
        # text_polarity = torch.unsqueeze(self.text_sentiment_polarity(text)[0],dim=1) #[32,1,200] #对于句子
        # #print("text_base, text_polarity ", type(text_base.size), text_polarity.size())
        # text_base = torch.cat((text_base, text_polarity), dim=-1) #[32,1,712]
        # text_attention,_ = self.attention_text( text_base, text_trans, text_base) #[32,1,712] Q=text_base, K=text_trans,V=text_base
        # text_attention = torch.squeeze(text_attention) #[32,712]
        #
        # # ------------------------------------- attribution ---------------------------------------
        # attribute_trans = self.TransformerEncoder_attribution(attribute_seq) #[32, 5, 200]
        # attribute_base,_ = self.base_attribution(attribute_seq) #[32, 5, 200]
        # #print("attribute_base ",attribute_seq.size(), attribute_trans.size(), attribute_base.size())
        # s0 = attribute_base.size(0)
        # s1 = attribute_base.size(1)
        # s2 = attribute_base.size(2)
        # attribute_base = attribute_base.contiguous().view(s0, s1*s2) #[32,1000]
        # attribute_polarity = self._attribute_sentiment_polarity(attribute_words) #[32,5]
        # attribute_base = self.attr_linear( torch.cat((attribute_base, attribute_polarity), dim=-1) )#[32,1005] -> [32,1000]
        # attribute_base = attribute_base.contiguous().view(s0, s1, s2) #[32, 5, 200]
        # attribute_attention,_ = self.attention_attribution(attribute_base, attribute_trans, attribute_base) #[32, 5, 200] Q=attribute_base,K=attribute_trans, V=attribute_base
        #
        # # ------------------------------------- image_seq ---------------------------------------
        # image_trans = self.ViTTransformerEncoder_image(image_seq, mask=None) #[196, 32, 500]
        # image_base = self.base_image_seq(image_seq.permute(1,0,2)).permute(1,0,2) ##[196, 32, 125]
        # image_polarity = torch.unsqueeze( attribute_polarity ,dim=1) # #[32,1,5]
        # image_polarity = image_polarity.repeat(1,196,1)
        # image_polarity = image_polarity.permute(1,0,2)
        # image_base = torch.cat((image_base,image_polarity), dim=-1)
        # image_base = self.image_seq_linear(image_base) # [196, 32, 500]
        # image_seq_attention = self.attention_image_seq( image_base,image_trans, image_base) # [196, 32, 500] Q=image_base,K=image_trans, V=image_base
        #
        # # [32,200]  [32, 5]
        # sentiment_gap = polarity_gap(torch.squeeze(text_polarity), attribute_polarity, self.batchsize)
        # # 输入和返回都是[196, 32, 500]    [32, 5, 200]     [32,712]
        # image_seq, attribute_seq, text_result = self.Coattention(image_seq_attention, attribute_attention, text_attention ,sentiment_gap)
        #






        #------------------------------------------------------------------------------------------
        wt = self.wt_relu(self.wt)

        text_polarity = self.text_sentiment_polarity(text)[0]
        attribute_polarity = self._attribute_sentiment_polarity(attribute_words) #[32,5]

        image_result = self.images_linear(image_result)
        attribute_result = self.attribute_linear(attribute_result)
        text_result = self.text_linear(text_result)

        image_result_trans_out = torch.squeeze(self.sharetransformer(torch.unsqueeze(image_result, dim=1)),dim=1)
        attribute_result_trans_out = torch.squeeze(self.sharetransformer(torch.unsqueeze(attribute_result,dim=1)),dim=1)
        text_result_trans_out = torch.squeeze(self.sharetransformer(torch.unsqueeze(text_result,dim=1)), dim=1)

        #print("image_result attribute_result text_result",image_result.size(), attribute_result.size(), text_result.size())
        #print("image_result_trans_out ", image_result_trans_out.size(), attribute_result_trans_out.size(), text_result_trans_out.size())
        attribute_senti_polarity = self._attribute_sentiment_polarity(attribute_words)
        #print("image_result_trans_out, attribute_senti_polarity  ",image_result_trans_out.size(), attribute_senti_polarity.size() )
        image_result_trans_out_senti = torch.cat(( image_result_trans_out, attribute_senti_polarity ), dim=-1)
        attribute_result_trans_out_senti = torch.cat((attribute_result_trans_out, attribute_senti_polarity), dim=-1)
        #print("attribute_result_trans_out_senti ", text_result_trans_out.size(), text_polarity.size())
        text_result_trans_out_senti = torch.cat((text_result_trans_out, text_polarity), dim=-1)
        image_att = torch.cat((image_result_trans_out_senti, attribute_result_trans_out_senti), dim=-1)
        image_att = self.image_attr_linear(image_att)
        modality = torch.tanh( self.crossattention(text_result_trans_out_senti, image_att, image_att)[0] )
        # modality_FFW1_in = torch.cat((modality,text_result_trans_out_senti), dim=-1)
        modality_FFW1_in = modality + wt * text_result_trans_out_senti
        modality_FFW1_out = torch.tanh( self.FFW1(modality_FFW1_in) ) #256

        selfattention_out, _ = self.selfattention(modality_FFW1_out, modality_FFW1_out, modality_FFW1_out) #256
        modality_FFW2_in = torch.tanh(selfattention_out) #256
        modality_FFW2_out = self.FFW2(modality_FFW2_in) #256
        text_result = self.text_back_linear(modality_FFW2_out)

        # sentiment_gap = polarity_gap(torch.squeeze(text_polarity), attribute_polarity, self.batchsize)
        # modality_senti_gap = torch.cat((modality_FFW1_out, sentiment_gap), dim=-1)
        # selfattention_out, _ = self.selfattention(modality_senti_gap,modality_senti_gap,modality_senti_gap)
        #
        # modality_FFW2_in = torch.tanh( torch.cat((selfattention_out,image_result_trans_out), dim=-1) )
        # modality_FFW2_out = self.FFW2(modality_FFW2_in)
        # text_result = self.text_back_linear(modality_FFW2_out)

        image_result = self.image_back(image_result_trans_out_senti)
        attribute_result = self.attribute_back(attribute_result_trans_out_senti)

        return image_result, image_seq, attribute_result,attribute_seq, attribute_words, text, text_result


class Models(nn.Module):
    def __init__(self,batchsize,block_num, fc_dropout_rate,base_model = 'LSTM', base_model_layernum = 2, embedding_dim = 512,hidden_dim=100,output_dim=200,dim_feedforword=512,num_head=4,num_layers=2,dropout = 0.2,max_len=128):
        super(Models, self).__init__()
        self.block_num = block_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dim_feedforword = dim_feedforword
        self.num_head = num_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
        self.fc_dropout_rate = fc_dropout_rate
        self.base_model = base_model
        self.base_model_layernum = base_model_layernum
        self.batchsize = batchsize
        self.ViTTransformerEncoder_image = ViTTransformer(num_layers=6, dims=500, num_heads=8, ff_dim=512, dropout=0.2)
        self.attention_image_seq = MUSEAttention(d_model=500, d_k=512, d_v=512, h=8)

        if self.base_model == 'LSTM':
            self.base_text = nn.LSTM(input_size=embedding_dim, hidden_size=256,
                                     num_layers=base_model_layernum, bidirectional=True, dropout=dropout)
            self.base_attribution = nn.LSTM(input_size=200, hidden_size=hidden_dim[1], num_layers=base_model_layernum,
                                            bidirectional=True, dropout=dropout)
            # self.base_attribution = nn.Linear(200, base_model_layernum * hidden_dim[1])
            self.base_image_seq = nn.Conv1d(196, 196, 3, stride=4)
        elif self.base_model == 'GRU':
            self.base_text = nn.GRU(input_size=256, hidden_size=hidden_dim[0],
                                    num_layers=base_model_layernum, bidirectional=True, dropout=dropout)
            self.base_attribution = nn.GRU(input_size=200, hidden_size=hidden_dim[1], num_layers=base_model_layernum,
                                           bidirectional=True, dropout=dropout)
            # self.base_attribution = nn.Linear(200, base_model_layernum * hidden_dim[1])
            self.base_image_seq = nn.Conv1d(196, 196, 3, stride=4)
        elif self.base_model == 'Linear':
            self.base_text = nn.Linear(embedding_dim, base_model_layernum * 256)
            self.base_attribution = nn.Linear(200, base_model_layernum * hidden_dim[1])
            self.base_image_seq = nn.Conv1d(196, 196, 3, stride=4)
        else:
            print('Base model must be one of LSTM/GRU/Linear')
            raise NotImplementedError

        self.middlefuseblocks = nn.ModuleList([middlefuseblock(self.batchsize,self.fc_dropout_rate,base_model = 'LSTM', base_model_layernum = self.base_model_layernum, \
                                                               embedding_dim = 512,hidden_dim=hidden_dim,output_dim=200,dim_feedforword=512,num_head=4,num_layers=2,dropout = 0.2,max_len=128) for _ in range(block_num)])
    def forward(self, image_result, image_seq, text, text_result,text_seq, attribute_result, attribute_seq, attribute_words):
                    #  [32, 500]   [32,x]   [32, 512]    [32, 200]                  [5,32]真实单词
        for block in self.middlefuseblocks:
            image_result, image_seq, attribute_result,attribute_seq, attribute_words, text, text_result = \
                block(image_result, image_seq, attribute_result, attribute_seq, attribute_words, text, text_result)
        #mid_out = torch.cat((middlefuseblocks_image, middlefuseblocks_attribute, middlefuseblocks_text), dim=-1)
        #print("block over")
        #image_seq = self.ViTTransformerEncoder_image(image_seq)
        image_seq = self.attention_image_seq(image_seq, image_seq, image_seq)
        text_seq, _ = self.base_text(text_seq)
        attribute_seq,_ = self.base_attribution(attribute_seq)

        return image_result, image_seq, attribute_seq, attribute_words, text, text_result, text_seq







#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
class towerblock(nn.Module):
    def __init__(self):
        super(towerblock, self).__init__()

        self.trans_img = TowertransformerEncoder(num_layers=6, dims=500+5, dims2=512+300, dim_out=500, num_heads=5, ff_dim=512, dropout=0.2)
        self.trans_attr = TowertransformerEncoder(num_layers=6, dims=200+5, dims2=512+300, dim_out=200, num_heads=5, ff_dim=512, dropout=0.2)
        self.trans_text = TowertransformerEncoder(num_layers=6, dims=512+300, dims2=710, dim_out=512, num_heads=4, ff_dim=512, dropout=0.2)


    def forward(self, image_result,attribute_result,attribute_polarity, text_result, text_polarity):
        image_result = torch.cat(( image_result, attribute_polarity), dim=-1)   #dim=500+5
        attribute_result = torch.cat((attribute_result, attribute_polarity), dim=-1)  # dim=200+5
        text_result = torch.cat((text_result, text_polarity), dim=-1)  # dim=512+200
        imgout, imgout_hid = self.trans_img(image_result, text_result)
        attrout, attrout_hid = self.trans_attr(attribute_result, text_result)
        textout, textoutt_hid = self.trans_text(text_result, torch.cat((image_result, attribute_result), dim=-1))

        #        505      505        205       205         712       712
        return imgout, imgout_hid, attrout, attrout_hid, attribute_polarity, textout, textoutt_hid, text_polarity
        #return imgout, attrout, attribute_polarity, textout, text_polarity


class memory_retrive_sentiment(nn.Module):
    def __init__(self, modality, hop=3):
        super(memory_retrive_sentiment, self).__init__()
        self.hop = hop
        self.modality = modality
        self.drop = nn.Dropout(0.2)
        self.w = nn.Parameter(torch.Tensor(1))

        if modality == 'i':
            self.alig_linear = nn.Linear(505,500)
            #self.attention = nn.MultiheadAttention(embed_dim=500, num_heads=1)
            self.attention =  ScaledDotProductAttention(d_model=500, d_k=500, d_v=500, h=4)
            self.projection = nn.Sequential(
                nn.Linear(500,500),
                torch.nn.ReLU(),
                nn.Linear(500,500),
            )
            self.momery_alig =  nn.Linear(500+1000,500)
        elif modality == 'a':
            self.alig_linear = nn.Linear(205, 200)
            self.attention = nn.MultiheadAttention(embed_dim=200, num_heads=1)
            self.attention = self.attention =  ScaledDotProductAttention(d_model=200, d_k=200, d_v=200, h=4)
            self.projection = nn.Sequential(
                nn.Linear(200, 200),
                torch.nn.ReLU(),
                nn.Linear(200, 200),
            )
            self.momery_alig =  nn.Linear(200+1000, 200)
        else:
            self.alig_linear = nn.Linear(712, 512)
            self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=1)
            self.attention = self.attention =  ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=4)
            self.projection = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
            self.momery_alig = nn.Linear(512+1000, 512)


    def forward(self, momery, hid, text_polarity, attribute_polarity):


        #with torch.no_grad():

        #sentiment_gap = polarity_gap(torch.squeeze(text_polarity, dim=0), attribute_polarity)
        #hid = self.drop(self.alig_linear(torch.cat((hid, sentiment_gap), dim=-1)))  # 此时，momery和hid最后维度一致, momery是三维，  hid是二维

        #sentiment_gap = sentiment_gap.repeat(momery.size(0), 1, 1)
        #momery = self.momery_alig( torch.cat((momery, sentiment_gap), dim=-1) )

        #hid = self.drop(self.alig_linear(hid))
        length = momery.size(0)  # memory [196,32,500] hid [32,500]
        hid = torch.unsqueeze(hid,dim=1)
        for i in range(self.hop):
            temp = torch.ones_like(momery)
            momerymid = self.projection(momery)

            for j in range(length):
                attinkv = hid
                momeryq = torch.unsqueeze(momerymid[j],dim=1)
                q = self.attention(momeryq, attinkv, attinkv)
                temp[j] = torch.squeeze( q, dim=1)
            #print("momery ",momery.size())
            momery = momery + self.w * temp
        return momery

class towertransformers(nn.Module):
    def __init__(self,batchsize,block_num, fc_dropout_rate,base_model = 'LSTM', base_model_layernum = 2, embedding_dim = 512,hidden_dim=100,output_dim=200,dim_feedforword=512,num_head=4,num_layers=2,dropout = 0.2,max_len=128):
        super(towertransformers, self).__init__()
        self.block_num = block_num #这里设为1
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dim_feedforword = dim_feedforword
        self.num_head = num_head
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
        self.fc_dropout_rate = fc_dropout_rate
        self.base_model = base_model
        self.base_model_layernum = base_model_layernum
        self.batchsize = batchsize
        self.ViTTransformerEncoder_image = ViTTransformer(num_layers=6, dims=500, num_heads=8, ff_dim=512, dropout=fc_dropout_rate)
        self.attention_image_seq = MUSEAttention(d_model=500, d_k=512, d_v=512, h=8)

        self.img_attention = nn.MultiheadAttention(embed_dim=500, num_heads=4)
        self.attri_attention = nn.MultiheadAttention(embed_dim= 200, num_heads=4)
        self.text_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
        self.img_FFW = PositionWiseFeedForward_change_dim(500, 512, 500)
        self.attri_FFW = PositionWiseFeedForward_change_dim(200, 512, 200)
        self.text_FFW = PositionWiseFeedForward_change_dim(embedding_dim, 256, 512)

        self.wi = nn.Parameter(torch.Tensor(1))
        self.wa = nn.Parameter(torch.Tensor(1))
        self.wt = nn.Parameter(torch.Tensor(1))
        self.temperature = nn.Parameter(torch.Tensor(1))

        if self.base_model == 'LSTM':
            self.base_text = nn.LSTM(input_size=embedding_dim, hidden_size=256,
                                     num_layers=base_model_layernum, bidirectional=True, dropout=fc_dropout_rate)
            self.base_attribution = nn.LSTM(input_size=200, hidden_size=hidden_dim[1], num_layers=base_model_layernum,
                                            bidirectional=True, dropout=fc_dropout_rate)
            # self.base_attribution = nn.Linear(200, base_model_layernum * hidden_dim[1])
            self.base_image_seq = nn.Conv1d(196, 196, 3, stride=4)
        elif self.base_model == 'GRU':
            self.base_text = nn.GRU(input_size=256, hidden_size=hidden_dim[0],
                                    num_layers=base_model_layernum, bidirectional=True, dropout=fc_dropout_rate)
            self.base_attribution = nn.GRU(input_size=200, hidden_size=hidden_dim[1], num_layers=base_model_layernum,
                                           bidirectional=True, dropout=fc_dropout_rate)
            # self.base_attribution = nn.Linear(200, base_model_layernum * hidden_dim[1])
            self.base_image_seq = nn.Conv1d(196, 196, 3, stride=4)
        elif self.base_model == 'Linear':
            self.base_text = nn.Linear(embedding_dim, base_model_layernum * 256)
            self.base_attribution = nn.Linear(200, base_model_layernum * hidden_dim[1])
            self.base_image_seq = nn.Conv1d(196, 196, 3, stride=4)
        else:
            print('Base model must be one of LSTM/GRU/Linear')
            raise NotImplementedError

        self.text_sentiment_polarity = TextFeature.sentiment_polarity(batch_size=batchsize)
        self._attribute_sentiment_polarity = attribute_sentiment_polarity(batchsize=batchsize)

        #self.towerblock = towerblock()
        self.towerfuse_module = nn.ModuleList([
                towerblock() for _ in range(block_num)
            ])
        #self.towerfuse_module = towerblock()

        self.memory_retrive_img = memory_retrive_sentiment( modality='i', hop = 3)
        self.memory_retrive_attr = memory_retrive_sentiment( modality= 'a',hop = 3)
        self.memory_retrive_text = memory_retrive_sentiment( modality='t', hop = 3)

        self.text_polarity_aware = nn.Linear( 300, 512)
        self.attribute_polarity_aware = nn.Linear( 5, 200)
        self.image_polarity_aware = nn.Linear(5, 500)

        self.text_sentiment_attention_itc = nn.Linear(500, 200)
        self.text_embed_itc = nn.Linear(512, 200)
        self.image_embed = nn.Linear(512, 200)

    #计算itc_loss
    def sentiment_aware_contrast_learning_loss(self, text_sentiment_attention, text_embed, image_embed):

        image_feat = F.normalize(self.text_sentiment_attention_itc(image_embed), dim=1)
        text_feat = F.normalize(self.text_embed_itc(text_embed), dim=1)
        text_sentiment_feat = F.normalize(self.image_embed(text_sentiment_attention), dim=1)
        sim1 = image_feat@text_feat.t()/self.temperature
        sim2 = text_sentiment_feat @ image_feat.t()/self.temperature
        itc_loss1 = sim1.diag().sum()
        itc_loss2 = sim2.diag().sum()
        itc_sum = itc_loss1 + itc_loss2
        sm = torch.exp(itc_loss2)/torch.exp(itc_sum)
        return -F.logsigmoid(sm)



    def forward(self, image_result, image_seq, text, text_result, text_seq, attribute_result, attribute_seq, attribute_words):
        #用于传入下一个模块
        #print("image_result_0 ", image_result.size())
        text_result_itc = text_result
        image_result_itc = image_result
        text_polarity = self.text_sentiment_polarity(text)[0]  # [32,200]
        attribute_polarity = self._attribute_sentiment_polarity(attribute_words)  # [32,5]

        text_polarity_ = self.text_polarity_aware(text_polarity)
        attribute_polarity_ = self.attribute_polarity_aware(attribute_polarity)
        image_polarity_ = self.image_polarity_aware(attribute_polarity)

        image_result, _ = self.img_attention(image_polarity_,image_result,image_result)
        image_result = self.img_FFW(image_result) + self.wi * image_polarity_
        attribute_result, _ = self.attri_attention(attribute_polarity_,attribute_result,attribute_result)
        attribute_result = self.attri_FFW(attribute_result) + self.wa * attribute_polarity_
        text_result, _ = self.text_attention(text_polarity_, text_result,text_result)
        text_result = self.text_FFW(text_result) + self.wt * text_polarity_

        itc_loss = self.sentiment_aware_contrast_learning_loss(text_result, text_result_itc, image_result_itc)

    # imgout, imgout_hid, attrout, attrout_hid, textout, textoutt_hid
    #     for block in self.towerfuse_module:
    #         image_result, imgout_hid, attribute_result, attrout_hid, text_result, textoutt_hid = \
    #             block(image_result, attribute_result, attribute_polarity, text_result, text_polarity)
    #     for block in self.towerfuse_module:
    #         image_result, attribute_result, attribute_polarity, text_result, text_polarity = \
    #             block(image_result, attribute_result, attribute_polarity, text_result, text_polarity)
    #     image_result, imgout_hid, attribute_result, attrout_hid, text_result, textoutt_hid = \
    #             self.towerfuse_module(image_result, imgout_hid, attribute_result,attrout_hid, attribute_polarity, text_result,textoutt_hid, text_polarity)

        for block in self.towerfuse_module:
            image_result, imgout_hid, attribute_result, attrout_hid, attribute_polarity, text_result, textoutt_hid, text_polarity = \
                block(image_result,  attribute_result, attribute_polarity,
                                  text_result, text_polarity)

        #         500      505        200       205         512       712
        #return imgout, imgout_hid, attrout, attrout_hid, textout, textoutt_hid

        image_seq = self.attention_image_seq(image_seq, image_seq, image_seq)
        #image_seq = self.ViTTransformerEncoder_image(image_seq)
        text_seq, _ = self.base_text(text_seq)
        attribute_seq, _ = self.base_attribution(attribute_seq)
        #print(" hid!!!!",imgout_hid.size(), attrout_hid.size(),textoutt_hid.size())
        #imgout_hid = torch.squeeze(imgout_hid, dim=0)
        #attrout_hid = torch.squeeze(attrout_hid, dim=0)
        #textoutt_hid = torch.squeeze(textoutt_hid, dim=0)

        # image_seq = self.memory_retrive_img(image_seq, image_result, text_polarity, attribute_polarity)
        # attribute_seq = self.memory_retrive_attr(attribute_seq, attribute_result, text_polarity, attribute_polarity)
        # text_seq = self.memory_retrive_text(text_seq, text_result, text_polarity, attribute_polarity)




        return image_result, image_seq, attribute_seq, attribute_result, attribute_words, text, text_result, text_seq, itc_loss


if __name__ == "__main__":
    image=ImageFeature.ExtractImageFeature()
    textfeature=TextFeature.ExtractTextFeature(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN)
    attribute=AttributeFeature.ExtractAttributeFeature()
    fuse=ModalityFusion()
    for text,text_index,image_feature,attribute_index,group,id in LoadData.train_loader:
        image_result,image_seq=image(image_feature)
        text_result,text_seq=textfeature(text)
        attribute_result,attribute_seq=attribute(attribute_index)
        result=fuse(image_result,image_seq,text_result,text_seq.permute(1,0,2),attribute_result,attribute_seq.permute(1,0,2))
        print(result.shape)  #[32, 512]
        print(result)

        break