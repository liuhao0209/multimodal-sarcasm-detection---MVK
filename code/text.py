import torch


# a = torch.tensor([[1,1,2],[1,2,3],[2,2,3]])
# b = torch.ones_like(a)
# c = a+b
# d = torch.mul(a, c)
# print(d)

import numpy as np
a = torch.from_numpy( np.arange(15).reshape(3,5) )
b = torch.from_numpy( np.arange(15).reshape(3,5) )

sim1 = b@a.t()

sim2 = a@b.t()

loss1 = sim1.diag().sum()
loss2 = sim2.diag().sum()
print("----  ", loss1, loss2)




a = torch.randn(1, 512, requires_grad=True)
print("--------1 ", a.size(), type(a), a.dtype)
b = a.transpose(0,1)
print("--------1 ", b.size(), type(b), b.dtype)