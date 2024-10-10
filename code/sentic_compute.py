#from senticnet7 import senticnet
import csv
import xlrd
import pandas as pd
y = 3
def load_sentic_word():
    """
    load senticNet
    """
    path = './senticNet/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = float(sentic)
    fp.close()
    return senticNet

# senticNet = load_sentic_word()

# def get_sentic_score(word_i,word_j):
#     if word_i not in senticNet or word_j not in senticNet or word_i == word_j:
#         return 0
#     return abs(float(senticNet[word_i] - senticNet[word_j])) * y**(-1*senticNet[word_i]*senticNet[word_j])

def load_sentic_7_word():
    """
    load senticNet
    """
    path = './senticNet/senticnet_7.txt'
    senticNet_7 = {}
    fp = open(path, 'r',encoding='utf-8')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        line_pair = line.split()
        word, sentic = line_pair[0], line_pair[1]
        senticNet_7[word] = float(sentic)
    fp.close()
    return senticNet_7

def load_related_word():
    pathDATA = r'G:\program_code_data\senticnet\senticnet7_related.csv'

    with open(pathDATA, 'r') as f:
        reader = csv.DictReader(f)
        raw = [cow['raw'] for cow in reader]

    with open(pathDATA, 'r') as f:
        reader = csv.DictReader(f)
        related_word1 = [cow['related_word1'] for cow in reader]

    with open(pathDATA, 'r') as f:
        reader = csv.DictReader(f)
        related_word2 = [cow['related_word2'] for cow in reader]

    with open(pathDATA, 'r') as f:
        reader = csv.DictReader(f)
        related_word3 = [cow['related_word3'] for cow in reader]

    with open(pathDATA, 'r') as f:
        reader = csv.DictReader(f)
        related_word4 = [cow['related_word4'] for cow in reader]

    with open(pathDATA, 'r') as f:
        reader = csv.DictReader(f)
        related_word5 = [cow['related_word5'] for cow in reader]

    return raw, related_word1, related_word2, related_word3, related_word4, related_word5


def generate_related():
    file_related = r'G:\program_code_data\senticnet\senticnet1.xlsx'
    fp = pd.read_excel(file_related,usecols=[0,9,10,11,12,13])
    data = fp.values #149290
    print("data ", type(data),data.shape,data)
    raw_word = []
    related_word1 = []
    related_word2 = []
    related_word3 = []
    related_word4 = []
    related_word5 = []
    for i in range(149290):
        raw_word.append(data[i][0])
        related_word1.append(data[i][1])
        related_word2.append(data[i][2])
        related_word3.append(data[i][3])
        related_word4.append(data[i][4])
        related_word5.append(data[i][5])

    pathDATA = r'G:\program_code_data\senticnet\senticnet7_related.csv'

    dataframe = pd.DataFrame({'raw': raw_word, 'related_word1': related_word1, 'related_word2': related_word2, 'related_word3': related_word3, 'related_word4': related_word4, 'related_word5': related_word5})
    dataframe.to_csv(pathDATA, index=False, sep=",")


if __name__ == "__main__":
    sheet_names = load_sentic_7_word()
    print("sheet_names", type(sheet_names),len(sheet_names))
    print(type(sheet_names['zombification']))
    #generate_related()
