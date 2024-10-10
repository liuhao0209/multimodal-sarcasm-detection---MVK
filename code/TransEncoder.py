import transformers
import torch
import torch.nn as nn
import math

device = torch.device('cuda:0')

class PositonalEncoding(nn.Module):
    def __init__(self,d_model,dropout = 0.2,max_len=512):
        super(PositonalEncoding, self).__init__()

        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model) )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position *div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

def length_to_mask(lengths):
    '''
    length = torch.tensor([])
    length_to_mask(lengths
    :param lengths: [batch,]
    :return: batch*max_len
    '''
    max_len = torch.max(lengths)
    max_len_cuda = torch.arange(max_len).to(device)
    mask = max_len_cuda.expand(lengths.shape[0], max_len).to(device) < lengths.unsqueeze(1).to(device)
    return mask


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=100, output_dim=512,dim_feedforword=512,num_head=4,num_layers=2,dropout = 0.2,max_len=1024,activate:str="relu"):
        super(TransformerEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        #self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.position_embedding = PositonalEncoding(embedding_dim,dropout,max_len)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim,num_head,dim_feedforword,dropout,activate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.transformer_out = nn.Linear(hidden_dim,output_dim)

    def forward(self, inputs):
        hidden_states = self.position_embedding(inputs)
        #attention_mask = length_to_mask(lengths) == False
        #print("attention_mask ",attention_mask.size())
        hidden_states = self.transformer(hidden_states)


        #hidden_states = hidden_states[0, :, :]
        out = self.transformer_out(hidden_states)
        return out


if __name__ == '__main__':
    trans = TransformerEncoder(embedding_dim = 200,hidden_dim=200,output_dim=200,dim_feedforword=512,num_head=4,num_layers=2,dropout = 0.2,max_len=1024)
    inputs = torch.randn(32,5,200)
    lens = torch.tensor([53, 27, 59, 8])
    print("input ",inputs.size())
    out = trans(inputs).to(device)
    print("out ",out.size())


