import torch
import torch.nn as nn

class CoAttentionlayer(nn.Module):
    def __init__(self,embedding_dim):
        super(CoAttentionlayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.imte_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8)
        self.atrite_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8)
        self.sentiment_gap_linear = nn.Linear(embedding_dim + 1000,embedding_dim)

        self.w1 = nn.Parameter(torch.Tensor(1))
        self.w1_relu = nn.ReLU()

        self.image_convert = nn.Linear(500, embedding_dim)
        self.attribute_convert = nn.Linear(5*200,embedding_dim)

        self.image_convert_to_back = nn.Linear(embedding_dim,500)
        self.attribute_convert_to_back = nn.Linear(embedding_dim,5*200)

    def forward(self,image, attribution, text, sentiment_gap): # [196, 32, 500] [32, 5, 200] [32,512]  [32,1000]
        image_seq_list_1 = list()
        attribute_list_2 = list()
        output_list_3 = list()
        w1 = self.w1_relu(self.w1)
        #text_sentiment = torch.cat((text), dim = -1)
        #attribution = attribution.permute(1,0,2) #[5, 32, 200]
        s0 = attribution.size(0)
        s1 = attribution.size(1)
        s2 = attribution.size(2)
        attribution = attribution.contiguous().view(s0, s1*s2) ##[32,1000]
        image = self.image_convert(image)
        attribution = self.attribute_convert(attribution) #[ 32, 5, 512]
        for i in range(image.size(0)):
            image_seq, _ = self.imte_attn(text,image[i],text) #q=text,k=image[i],v=text
            image_seq_list_1.append(image_seq)
        image_seq = torch.torch.stack(image_seq_list_1)
        attribute_result,_ = self.atrite_attn(text, attribution, text)

        text_temp = torch.cat((text,sentiment_gap),dim=-1)
        text_result = w1 * self.sentiment_gap_linear(text_temp)

        image_seq = self.image_convert_to_back(image_seq)
        #print("attribute_result ",type(attribute_result))
        attribute_result = self.attribute_convert_to_back(attribute_result)
        attribute_result = attribute_result.contiguous().view(s0, s1, s2)

        return image_seq, attribute_result, text_result  #[196, 32, 500]    [32, 5, 200]     [32,512]

