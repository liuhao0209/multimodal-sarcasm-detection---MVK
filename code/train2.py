import torch
import ImageFeature
import AttributeFeature
import TextFeature
import FinalClassifier
import FuseAllFeature
from LoadData import *
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Multimodel(torch.nn.Module):
    def __init__(self, lstm_dropout_rate, fc_dropout_rate):
        super(Multimodel, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.attribute = AttributeFeature.ExtractAttributeFeature()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN)
        self.fuse = FuseAllFeature.ModalityFusion()
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)

    def forward(self, text_index, image_feature, attribute_index):
        image_result, image_seq = self.image(image_feature)
        attribute_result, attribute_seq = self.attribute(attribute_index)
        text_result, text_seq = self.text(text_index, attribute_result)
        fusion = self.fuse(image_result, image_seq, text_result, text_seq.permute(1, 0, 2), attribute_result,
                           attribute_seq.permute(1, 0, 2))
        output = self.final_classifier(fusion)
        return output


def train(model,train_loader,val_loader,loss_fn,optimizer,number_of_epoch):
    for epoch in range(number_of_epoch):
        train_loss=0
        correct_train=0
        model.train()
        for text_index, image_feature, attribute_index, group, id in train_loader:
            group = group.view(-1,1).to(torch.float32).to(device)
            pred = model(text_index.to(device), image_feature.to(device), attribute_index.to(device))
            loss = loss_fn(pred, group)
            train_loss+=loss
            correct_train+=(pred.round()==group).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate valid loss

        valid_loss=0
        correct_valid=0
        model.eval()
        with torch.no_grad():
            for val_text_index, val_image_feature, val_attribute_index, val_group, val_id in val_loader:
                val_group = val_group.view(-1,1).to(torch.float32).to(device)
                val_pred = model(val_text_index.to(device), val_image_feature.to(device), val_attribute_index.to(device))
                val_loss = loss_fn(val_pred, val_group)
                valid_loss+=val_loss
                correct_valid+=(val_pred.round()==val_group).sum().item()

        print("epoch: %d train_loss=%.5f train_acc=%.3f valid_loss=%.5f valid_acc=%.3f"%(epoch,
                                                                                         train_loss/len(train_loader),
                                                                                      correct_train/len(train_loader)/batch_size,
                                                                                         valid_loss/len(val_loader),
                                                                                         correct_valid/len(val_loader)/batch_size))



learning_rate_list = [0.001]
fc_dropout_rate_list=[0,0.3,0.9,0.99]
lstm_dropout_rate_list=[0, 0.2, 0.4]
weight_decay_list=[0,1e-6,1e-5,1e-4]
# weight_decay_list=[1e-7]
batch_size=128
data_shuffle=False

# load data
train_fraction=0.8
val_fraction=0.1
train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=data_shuffle)
val_loader = DataLoader(val_set,batch_size=batch_size, shuffle=data_shuffle)
test_loader = DataLoader(test_set,batch_size=batch_size, shuffle=data_shuffle)
play_loader = DataLoader(test_set,batch_size=1, shuffle=data_shuffle)

# start train
import itertools
comb = itertools.product(learning_rate_list, fc_dropout_rate_list,lstm_dropout_rate_list,weight_decay_list)
for learning_rate, fc_dropout_rate,lstm_dropout_rate,weight_decay in list(comb):
    print(f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")
    # loss function
    loss_fn=torch.nn.BCELoss()
    # initilize the model
    model = Multimodel(lstm_dropout_rate,fc_dropout_rate).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    # train
    number_of_epoch=7
    train(model,train_loader,val_loader,loss_fn,optimizer,number_of_epoch)


import sklearn.metrics as metrics
import seaborn as sns
def validation_metrics (model, dataset):
    model.eval()
    with torch.no_grad():
        correct=0
        confusion_matrix_sum=None
        loss_sum=0
        for text_index, image_feature, attribute_index, group, id in dataset:
            group = group.view(-1,1).to(torch.float32).to(device)
            pred = model(text_index.to(device), image_feature.to(device), attribute_index.to(device))
            loss = loss_fn(pred, group)
            loss_sum+=loss
            correct+=(pred.round()==group).sum().item()
            # calculate confusion matrix
            if confusion_matrix_sum is None:
                confusion_matrix_sum=metrics.confusion_matrix(group.to("cpu"),pred.round().to("cpu"),labels=[0,1])
            else:
                confusion_matrix_sum+=metrics.confusion_matrix(group.to("cpu"),pred.round().to("cpu"),labels=[0,1])
        acc=correct/len(dataset)/batch_size
        loss_avg=loss_sum/len(dataset)
    return loss_avg.item(), acc, confusion_matrix_sum

def plot_confusion_matrix(confusion_matrix):
    emotions=['not sarcasm','sarcasm']
    sns.heatmap(confusion_matrix, annot=True, xticklabels=emotions, yticklabels=emotions, fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
loss, acc, confusion_matrix=validation_metrics (model, test_loader)
print("loss:",loss,"accuracy:",acc)
plot_confusion_matrix(confusion_matrix)


import matplotlib.pyplot as plt
def validation_metrics (model, dataset):
    model.eval()
    with torch.no_grad():
        count=0
        for text_index, image_feature, attribute_index, group, id in dataset:
            if count==5:
                break
            id=id.item()
            print(f">>>Example {count+1}<<<")
            img=all_Data.image_loader(id)
            plt.imshow(img.permute(1,2,0))
            plt.show()
            print("Text: ",all_Data.text_loader(id))
            print("Labels: ",all_Data.label_loader(id))
            print(f"Truth:{' not ' if group[0]==0 else ' '}sarcasm")
            pred = model(text_index.to(device), image_feature.to(device), attribute_index.to(device))
            print(f"Preduct:{' not ' if round(pred[0,0].item())==0 else ' '}sarcasm")
            count+=1

validation_metrics (model, play_loader)







import torch
import torch.nn as nn
import time
import datetime
import ImageFeature
import AttributeFeature
import TextFeature
import FinalClassifier
import FuseAllFeature
from LoadData import *
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from ViT_Transfomer import ViTTransformer, PositionWiseFeedForward
from dateutil import tz
import matplotlib.pyplot as plt
#from AttributeFeature import word_convert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import MultiView_Module
# In[2]:
import random
'''
def seed_everything(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
'''
seed_random = 2023

class Multimodel(torch.nn.Module):
    def __init__(self, fc_dropout_rate, Multi_Nums, batchsize):
        super(Multimodel, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        self.attribute = AttributeFeature.ExtractAttributeFeature()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH, TEXT_HIDDEN)
        self.fuse = FuseAllFeature.ModalityFusion(image_dim=500, text_dim=512, attribute_dim=512)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate)
        self.Multiview = MultiView_Module.MultiviewFusion(dims=512, image_dim=500, text_dim=512, attribute_dim=512, dropout_rate_m = fc_dropout_rate, Multilayer_Nums=Multi_Nums)
        '''
        self.towerfuse = FuseAllFeature.towertransformers(batchsize, block_num=2, fc_dropout_rate=fc_dropout_rate,
                                              base_model='LSTM', base_model_layernum=2, embedding_dim=512,
                                              hidden_dim=[156, 100], output_dim=200, dim_feedforword=512, num_head=4,
                                              num_layers=2, dropout=0.2, max_len=1024)
        # block_num中间层的层数，base_model_layernum是LSTM的层数，固定为2，embedding_dim是文本的维度，固定位512
        self.setifuse = FuseAllFeature.Models(batchsize, block_num=2, fc_dropout_rate=fc_dropout_rate,
                                              base_model='GRU', base_model_layernum=2, embedding_dim=512,
                                              hidden_dim=[156, 100], output_dim=200, dim_feedforword=512, num_head=4,
                                              num_layers=2, dropout=0.2, max_len=1024)

        '''
        self.l_linear = nn.Linear(1212 ,512)
        self.relu = nn.ReLU()

    def forward(self, text, image_feature, attribute_index, attribute_words):  # 查看传入模型的数据类型 text是32句话，image_feature是处理好的32个图片的数据， attribute_index
        image_result, image_seq = self.image(image_feature)  # 第一个是计算第二个在0维度的均值
        text_result, text_seq = self.text(text)  # text是真实的句子
        attribute_result, attribute_seq = self.attribute(attribute_index)
        image_result, text_result, attribute_result = self.Multiview(image_result, text_result, attribute_result)

        fusion = self.fuse(image_result, image_seq, text_result, text_seq.permute(1, 0, 2), attribute_result, attribute_seq.permute(1, 0, 2)) #原来的
        #fusion = self.relu(self.l_linear(torch.concat((image_result, text_result, attribute_result), dim=-1)))
        #print("-----  ", fusion.size())
        output = self.final_classifier(fusion)  # 和改这个


        '''
        image_result, image_seq, attribute_seq, attribute_result, attribute_words, text, text_result, text_seq, loss_itc = \
            self.towerfuse(image_result, image_seq, text, text_result, text_seq.permute(1, 0, 2), attribute_result, attribute_seq.permute(1, 0, 2),
                          attribute_words)
        fusion = self.fuse(image_result, image_seq, text_result, text_seq, attribute_result, attribute_seq)
        '''

        return output


# In[3]:
start_local_time = datetime.datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M")
print("start_time ", start_local_time)
start_time = time.time()


def train(model, train_loader, val_loader, loss_fn, optimizer, number_of_epoch, seed_random):
    for epoch in range(number_of_epoch):
        # print("epoch")
        train_loss = 0
        correct_train = 0
        model.train()  # model.train()的作用是启用Batch Normalization 和Dropout。 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。 model.train()是保证BN层能够用到每一批数据的均值和方差。
        #seed_everything(seed_random)
        for text, text_index, image_feature, attribute_index, attribute_words, group, id in train_loader:
            # text (32,句子) image_feature [32, 196, 1000]  attribute_index [32, 5] attribute_index代表attribute的特征
            # print(id)
            group = group.view(-1, 1).to(torch.float32).to(device)  # 真实值

            # pred = model(text, image_feature.to(device), attribute_index.to(device))
            pred = model(text, image_feature.to(device), attribute_index.to(device), attribute_words)
            loss = loss_fn(pred, group)
            train_loss = train_loss + loss
            correct_train += (pred.round() == group).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate valid loss

        valid_loss = 0
        correct_valid = 0
        positive_number = 0
        pred_positive_number = 0
        pred_right_positive_number = 0

        negtive_number = 0
        pred_negtive_number = 0
        pred_right_negtive_number = 0
        model.eval()
        with torch.no_grad():
            val_tatol = 0
            for val_text, val_text_index, val_image_feature, val_attribute_index, val_attribute_words, val_group, val_id in val_loader:
                val_group = val_group.view(-1, 1).to(torch.float32).to(device)
                val_pred = model(val_text, val_image_feature.to(device), val_attribute_index.to(device), val_attribute_words)
                val_loss = loss_fn(val_pred, val_group)
                valid_loss += val_loss
                correct_valid += (val_pred.round() == val_group).sum().item()

                val_tatol += len(val_pred)
                # Pre 精度是精确性的度量，表示被分为正例的示例中，实际为正例的比例：
                # Rec 灵敏度表示所有正例中，被分对的比例， 灵敏度与召回率Recall计算公式完全一致
                flag = torch.full_like(val_pred, -1).to(torch.float32)
                a = (val_pred.round() == 1).to(torch.float32)
                b = torch.where(a == 1, a, flag).round()
                positive_number += (val_group == 1).sum().item()
                pred_right_positive_number += (b == val_group).sum().item()
                pred_positive_number += (val_pred.round() == 1).sum().item()


                flag_neg = torch.full_like(val_pred, -1).to(torch.float32)
                c = (val_pred.round() == 1).to(torch.float32)
                d = torch.where(c != 1, c, flag_neg).round()
                negtive_number += (val_group == 0).sum().item()
                pred_negtive_number += (val_pred.round() == 0).sum().item()
                pred_right_negtive_number += (d == val_group).sum().item()



            P = pred_right_positive_number / pred_positive_number
            R = pred_right_positive_number / positive_number
            F = 2 * P * R / (P + R)



            P_neg = pred_right_negtive_number / pred_negtive_number
            R_neg = pred_right_negtive_number / negtive_number
            F_neg = 2 * P_neg * R_neg / (P_neg + R_neg)

            P_macro = (P + P_neg) / 2
            R_macro = (R + R_neg) / 2
            F_macro = (F + F_neg) / 2



        print(
            "epoch: %d train_loss=%.5f train_acc=%.5f valid_loss=%.5f valid_acc=%.5f Pre=%.5f Rec=%.5f F-score=%.5f" % (
            epoch,
            train_loss / len(train_loader),
            correct_train / len(train_loader) / batch_size,
            valid_loss / len(val_loader),
            correct_valid / val_tatol,
            P, R, F))

        with open("tiaocan.txt", "a") as file:
            file.write("\n")
            file.write("epoch: %d train_loss=%.5f train_acc=%.5f valid_loss=%.5f valid_acc=%.5f" % (epoch,
                                                                                                    train_loss / len(
                                                                                                        train_loader),
                                                                                                    correct_train / len(
                                                                                                        train_loader) / batch_size,
                                                                                                    valid_loss / len(
                                                                                                        val_loader),
                                                                                                    correct_valid / val_tatol))


# In[22]:


learning_rate_list = [0.0002]
fc_dropout_rate_list = [0.2]
# lstm_dropout_rate_list=[0, 0.2, 0.4]
# weight_decay_list=[1e-65,1e-6,1e-5,1e-4]  [1e-65]
weight_decay_list = [1e-65]
batch_size = 64
data_shuffle = True
weight = torch.tensor([0.86028]).to(device)
train_epochs = 10
arfa = 0.38
Multi_Nums = 2
# In[23]:


# load data
train_fraction = 0.8
val_fraction = 0.1
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=data_shuffle)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=data_shuffle)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=data_shuffle)
#play_loader = DataLoader(test_set, batch_size=1, shuffle=data_shuffle)
play_loader = test_loader
# In[ ]:
seed_random = 100
#seed_everything(seed_random)

# start train
import itertools

comb = itertools.product(learning_rate_list, fc_dropout_rate_list, weight_decay_list)
for learning_rate, fc_dropout_rate, weight_decay in list(comb):
    print(f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} |  weight decay={weight_decay} | Multi_Nums={Multi_Nums}")
    with open("tiaocan.txt", "a") as file:
        file.write("\n")
        file.write(f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} |  weight decay={weight_decay} | Multi_Nums={Multi_Nums}")
    # loss function
    #loss_fn = torch.nn.BCELoss()  # 这个可以换一下   nn.CrossEntropyLoss()
    loss_fn = torch.nn.BCELoss(weight=weight)
    # initilize the model
    model = Multimodel(fc_dropout_rate, Multi_Nums, batch_size).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # train
    number_of_epoch = train_epochs

    # priint parameter sum
    print("tatol number of model parameters: ", sum(x.numel() for x in model.parameters()))

    train(model, train_loader, val_loader, loss_fn, optimizer, number_of_epoch, seed_random=seed_random)

# In[14]:


import sklearn.metrics as metrics
import seaborn as sns


def validation_metrics(model, dataset):
    positive_number = 0
    pred_positive_number = 0
    pred_right_positive_number = 0

    negtive_number = 0
    pred_negtive_number = 0
    pred_right_negtive_number = 0
    model.eval()
    with torch.no_grad():
        correct = 0
        confusion_matrix_sum = None
        loss_sum = 0
        test_tatol = 0
        for text, text_index, image_feature, attribute_index, attribute_words, group, id in dataset:
            group = group.view(-1, 1).to(torch.float32).to(device)
            pred = model(text, image_feature.to(device), attribute_index.to(device), attribute_words)
            loss = loss_fn(pred, group)
            loss_sum += loss
            correct += (pred.round() == group).sum().item()

            test_tatol += len(pred)

            flag = torch.full_like(pred, -1).to(torch.float32)
            a = (pred.round() == 1).to(torch.float32)
            b = torch.where(a == 1, a, flag).round()
            positive_number += (group == 1).sum().item()
            pred_right_positive_number += (b == group).sum().item()
            pred_positive_number += (pred.round() == 1).sum().item() #预测的为1的所有数量

            flag_neg = torch.full_like(pred, -1).to(torch.float32)
            c = (pred.round() == 1).to(torch.float32)
            d = torch.where(c != 1, c, flag_neg).round()
            negtive_number += (group == 0).sum().item()
            pred_negtive_number += (pred.round() == 0).sum().item()
            pred_right_negtive_number += (d == group).sum().item()

            # calculate confusion matrix
            if confusion_matrix_sum is None:
                confusion_matrix_sum = metrics.confusion_matrix(group.to("cpu"), pred.round().to("cpu"), labels=[0, 1])
            else:
                confusion_matrix_sum += metrics.confusion_matrix(group.to("cpu"), pred.round().to("cpu"), labels=[0, 1])

        P = pred_right_positive_number / pred_positive_number
        R = pred_right_positive_number / positive_number
        F = 2 * P * R / (P + R)



        P_neg = pred_right_negtive_number / pred_negtive_number
        R_neg = pred_right_negtive_number / negtive_number
        F_neg = 2 * P_neg * R_neg / (P_neg + R_neg)

        P_macro = (P + P_neg) / 2
        R_macro = (R + R_neg) / 2
        F_macro = (F + F_neg) / 2



        #acc = correct / len(dataset) / batch_size
        acc = correct / test_tatol
        loss_avg = loss_sum / len(dataset)
    return loss_avg.item(), acc, P, R, F, P_macro, R_macro, F_macro, confusion_matrix_sum


def plot_confusion_matrix(confusion_matrix):
    emotions = ['not sarcasm', 'sarcasm']
    sns.heatmap(confusion_matrix, annot=True, xticklabels=emotions, yticklabels=emotions, fmt='g')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


loss, acc, P, R, F, P_macro, R_macro, F_macro, confusion_matrix = validation_metrics(model, test_loader)
print("----------test result----------")
print("loss:", loss, "accuracy:", acc, "Pre:", P, "Rec:", R, "F-score:", F, "Pre_macro", P_macro, "Rec_macro", R_macro, "F_macro", F_macro)
plot_confusion_matrix(confusion_matrix)

# code_run_time = round(time.time() - start_time, 2)
code_run_time = datetime.datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M")
print("code run time ", code_run_time)
a = learning_rate_list[0]
b = fc_dropout_rate_list[0]
with open("tiaocan.txt", "a") as f:
    f.write("\n")
    f.write(
        "learning_rate_list=%d fc_dropout_rate_list=%d train_epochs=%d loss=%.5f accuracy=%.5f Pre=%.5f Rec=%.5f F-score=%.5f Pre_macro=%.5f Rec_macro=%.5f F-score_macro=%.5f" % (
        a, b, train_epochs, loss, acc, P, R, F, P_macro, R_macro, F_macro))
    f.write("\n")
    f.write("start_local_time: %s " % start_local_time)
    f.write("run over time: %s " % code_run_time)
    f.write("---------------this program run over----------------" )
# In[21]:


'''
def play_metrics(model, dataset):
    model.eval()
    with torch.no_grad():
        count=0
        # text, text_index, image_feature, attribute_index, attribute_words, group, id in dataset:
        for text, text_index, image_feature, attribute_index, attribute_words, group, id in dataset:
            if count==5:
                break
            id=id.item()
            print(f">>>Example {count+1}<<<")
            img=all_Data.image_loader(id)
            plt.imshow(img.permute(1,2,0))
            plt.show()
            print("Text: ",all_Data.__text_loader(id))
            print("Labels: ",all_Data.label_loader(id))
            print(f"Truth:{' not ' if group[0]==0 else ' '}sarcasm")
            pred = model(text, image_feature.to(device), attribute_index.to(device))
            print(f"Preduct:{' not ' if round(pred[0,0].item())==0 else ' '}sarcasm")
            count+=1

play_metrics(model, play_loader)
'''

