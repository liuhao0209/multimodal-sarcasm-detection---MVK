import torch.nn as nn
import torch
m = nn.Conv1d(196, 196, 3, stride=4)
input = torch.randn(32, 196, 500)
#input = input.permute(0,2,1)
output = m(input)
print("output ",output.size())