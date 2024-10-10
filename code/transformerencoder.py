import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
#from torch_geometric.utils import softmax
#from torch_scatter import scatter_add
from torch.nn.utils.rnn import pad_sequence
from TransEncoder import TransformerEncoder


class TransformerModel(nn.Module):
    def __init__(self,args, input_size,classes,hidden_size,base_layer,dropout):
        super(TransformerModel, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True, dropout=dropout)
        self.TransformerEncoder = TransformerEncoder(embedding_dim=1024, hidden_dim=1024, output_dim=512, dim_feedforword=512, num_head=4, num_layers=2, dropout=0.2, max_len=1024)
        self.classify_layer = nn.Linear(512, classes)

    def forward(self, U,umask, qmask, seq_lengths):

        lengths = torch.tensor(seq_lengths)
        p3_x = self.TransformerEncoder(inputs=U, lengths=lengths)
        a = self.classify_layer(p3_x)
        U_cause = F.log_softmax(a, dim=-1)
        U_cause = torch.cat([U_cause[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return U_cause
