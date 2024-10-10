import pickle
import os
WORKING_PATH = r"./"




learning_rate_list = [0.001]
fc_dropout_rate_list=[0,0.3,0.9,0.99]
#lstm_dropout_rate_list=[0, 0.2, 0.4]
weight_decay_list=[0,1e-6,1e-5,1e-4]
import itertools
comb = itertools.product(learning_rate_list, fc_dropout_rate_list,weight_decay_list)
print("comb", len( list(comb)) )