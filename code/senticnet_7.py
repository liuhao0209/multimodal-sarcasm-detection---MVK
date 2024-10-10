import csv
import pandas as pd
import xlrd

readersentic7 = pd.read_excel("G:/program_code_data/senticnet/senticnet1.xlsx")
sheet = readersentic7.values

senticlist7 = {}
with open('./senticNet/senticnet_7.txt','w',encoding='utf-8') as f:
    for w in sheet:
        #senticlist7[w[0]] = w[7]
        if type(w[0]) == bool:
            continue
        else:
            f.writelines([str(w[0]), ' ', str(w[8]), '\n'])