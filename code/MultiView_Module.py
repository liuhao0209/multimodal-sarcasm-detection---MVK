import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

import submodules
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Multiviewlayer(nn.Module):
    def __init__(self, dims, dropout_rate):
        super(Multiviewlayer, self).__init__()
        self.weight_i = torch.randn( 1, dims, requires_grad=True).to(device)
        self.weight_t = torch.randn( 1, dims, requires_grad=True).to(device)
        self.ScaledDotProductAttention_t = submodules.ScaledDotProductAttention_image_text(temperature = np.power(512, 0.5), attn_dropout=dropout_rate)
        self.ScaledDotProductAttention_i = submodules.ScaledDotProductAttention_image_text(temperature= np.power(512, 0.5), attn_dropout=dropout_rate)
        self.PositionwiseFeedForward_t = submodules.PositionWiseFeedForward_change_dim(dim = dims *2 , ff_dim = dims *2 ,outdim = dims, dropout=dropout_rate)
        self.PositionwiseFeedForward_i = submodules.PositionWiseFeedForward_change_dim(dim = dims *2 , ff_dim = dims *2 ,outdim = dims, dropout=dropout_rate)
    def forward(self, image_feature, text_feature, attribute_feature):
        '''

        a = torch.randn(1, 512, requires_grad=True)
        print("--------1 ", a.size(), type(a), a.dtype)
        b = a.transpose(0,1)
        print("--------1 ", b.size(), type(b), b.dtype)
        '''
        text_feature_cat = torch.concat((text_feature, attribute_feature), dim=1)
        image_feature_cat = torch.concat((image_feature, attribute_feature), dim=1)
        text_F, att_t = self.ScaledDotProductAttention_t(text_feature_cat, self.weight_i, text_feature_cat) # torch.Size([128, 1, 512]) torch.Size([128, 1, 1])
        image_F, att_i = self.ScaledDotProductAttention_i(image_feature_cat, self.weight_t, image_feature_cat) # torch.Size([128, 1, 512]) torch.Size([128, 1, 1])
        text_f = att_i@text_feature_cat
        image_f = att_t@image_feature_cat
        text_Ff = self.PositionwiseFeedForward_t(torch.concat((text_F, text_f), dim=-1))
        image_Ff = self.PositionwiseFeedForward_i(torch.concat((image_F, image_f), dim=-1))
        image_feature, text_feature = image_Ff, text_Ff

        return image_feature, text_feature



class MultiviewFusion(nn.Module):
    def __init__(self, dims, image_dim, text_dim, attribute_dim, dropout_rate_m, Multilayer_Nums):
        super(MultiviewFusion, self).__init__()
        self.dims = dims
        self.dropout_rate = 0.4
        self.i_linear = nn.Linear(image_dim, dims, bias=True)
        self.a_linear = nn.Linear(attribute_dim, dims, bias=True)
        self.t_linear = nn.Linear(text_dim, dims, bias=True)

        self.Multiview = nn.ModuleList([Multiviewlayer(dims = dims, dropout_rate = self.dropout_rate) for _ in range(Multilayer_Nums)])

        self.ffd_image = submodules.PositionWiseFeedForward_change_dim(dim = dims , ff_dim = dims,outdim = image_dim, dropout=self.dropout_rate)
        self.ffd_text = submodules.PositionWiseFeedForward_change_dim(dim=dims, ff_dim=dims, outdim=text_dim, dropout=self.dropout_rate)
        self.ffd_attribute = submodules.PositionWiseFeedForward_change_dim(dim=dims, ff_dim=dims, outdim=attribute_dim, dropout=self.dropout_rate)

        self.sharedMHA1 = submodules.MultiHeadAttention(n_head=6, d_model=512, d_k=512, d_v=512, dropout=dropout_rate_m, is_regu=False)
        self.sharedMHA2 = submodules.MultiHeadAttention(n_head=6, d_model=512, d_k=512, d_v=512, dropout=dropout_rate_m, is_regu=False)
        self.sharedMHA3 = submodules.MultiHeadAttention(n_head=6, d_model=512, d_k=512, d_v=512, dropout=dropout_rate_m, is_regu=False)

    def forward(self, image_result, text_result, attribute_result):
        image_feature =  torch.unsqueeze(self.i_linear(image_result), dim=1)
        text_feature = torch.unsqueeze(self.t_linear(text_result), dim=1)
        attribute_feature = torch.unsqueeze(self.a_linear(attribute_result), dim=1)
        #image_feature, _ = self.sharedMHA1(image_feature,image_feature,image_feature)
        #text_feature, _ = self.sharedMHA2(text_feature, text_feature, text_feature)
        #attribute_feature, _   = self.sharedMHA3(attribute_feature, attribute_feature, attribute_feature)



        for Layer in self.Multiview:
            image_feature, text_feature = Layer(image_feature, text_feature, attribute_feature)

        image_result, text_result, attribute_result = \
            torch.squeeze(image_feature, dim=1), torch.squeeze(text_feature, dim=1), torch.squeeze(attribute_feature, dim=1)
        image_result = self.ffd_image(image_result)
        text_result = self.ffd_text(text_result)
        attribute_result = self.ffd_attribute(attribute_result)

        return image_result, text_result, attribute_result

if __name__ =='__main__':

    image_result = torch.randn(128, 500)
    text_result = torch.randn(128, 512)
    attribute_result = torch.randn(128, 200)
    Multiview = MultiviewFusion(dims=512, image_dim=500, text_dim=512, attribute_dim=200, dropout_rate = 0.2, Multilayer_Nums=3)
    image_result, text_result, attribute_result = Multiview(image_result, text_result, attribute_result)
    print("image_result, text_result, attribute_result ", image_result.size(), text_result.size(), attribute_result.size())





