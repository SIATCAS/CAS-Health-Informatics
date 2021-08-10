import numpy as np
# import pandas as pd
# import xlrd
# import matplotlib.pyplot as plt
from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from pandas import read_csv
# from sklearn.preprocessing import MinMaxScaler 
import torch,re,os, cv2
# from keras.utils import np_utils
# from sklearn.metrics import accuracy_score
# from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import LSTM,BatchNormalization,advanced_activations
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD,RMSprop,adam
from sklearn.utils import shuffle
import torch.nn as NN
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
import random 
from scipy import interpolate
import gc
import datetime

torch.__version__
torch.manual_seed(128)
np.random.seed(1024)

class CNN(NN.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = NN.Sequential(   
            NN.Conv1d(in_channels = 1, out_channels = 8, stride = 1 , kernel_size=5, padding=0),
            NN.BatchNorm1d(8), 
            NN.ReLU(),
            NN.MaxPool1d(1,2),
            NN.Dropout(0.5),
            NN.Conv1d(in_channels = 8, out_channels = 16, stride = 1 , kernel_size=4, padding=0),
            NN.BatchNorm1d(16), 
            NN.ReLU(),
            NN.MaxPool1d(1,2),
            NN.Dropout(0.5),
            NN.Conv1d(in_channels = 16, out_channels = 32, stride = 1 , kernel_size=3, padding=0),
            NN.BatchNorm1d(32), 
            NN.ReLU(),
            NN.MaxPool1d(1,2), 
            NN.Dropout(0.5),
            NN.Conv1d(in_channels = 32, out_channels = 64, stride = 1 , kernel_size=2, padding=0),
            NN.BatchNorm1d(64), 
            NN.ReLU(),
            NN.MaxPool1d(1,2), 
            NN.Dropout(0.5),
            NN.Conv1d(in_channels = 64, out_channels = 32, stride = 1 , kernel_size=2, padding=0),
            NN.BatchNorm1d(32), 
            NN.ReLU(),
            NN.MaxPool1d(1,2)
            )        
#            NN.Dropout(0.5))
#        self.avgpool = NN.AdaptiveAvgPool1d(output_size=(60))
        self.classifier = NN.Sequential(
                NN.Linear(736, 100),
                NN.Dropout(0.5),
                NN.Linear(100, 30),
                NN.Linear(30, 9))

    def forward(self, x):
        out = self.features(x)
#        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        #print(out.shape)
        out = self.classifier(out)
        return out



cnn = CNN()

cnn(torch.rand(1,1,750))

path1 = "G:/signal/new_time_cut"
path2 = "G:/signal/HBA1c.txt"
path3 = "G:/signal/paper_2/save"


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


clinical_data = np.loadtxt(path2)
clin_data = np.array(clinical_data)
ori_use_ID = clin_data[:,0].astype(int)
ori_HBA1c = clin_data[:,1]


data_dir_list1 = sorted(os.listdir(path1),key=numericalSort)

use_number = len(data_dir_list1)
clinical_len = len(ori_use_ID)
USE_ID_list = []
HBA1c_list = []
for i in range(use_number):
    temp_ID = int(data_dir_list1[i])
    for j in range(clinical_len):
        temp_cinical = ori_use_ID[j]
        if temp_ID == temp_cinical:
            USE_ID_list.append(temp_ID)
            HBA1c_list.append(ori_HBA1c[j])
            break
        
            
USE_ID_list = np.array(USE_ID_list) 
HBA1c_list = np.array(HBA1c_list) 
     
HBA1c_VALUE1 = HBA1c_list
G0 = 0
G1 = 0
G2 = 0
G3 = 0
G4 = 0
G5 = 0
G6 = 0
G7 = 0
G8 = 0
HBA1c_VALUE = np.zeros(len(HBA1c_VALUE1)) 
for t in range (len(HBA1c_VALUE1)):
    if HBA1c_VALUE1[t] <= 6.5:
        HBA1c_VALUE[t] = 0
        G0 = G0 + 1
    elif  HBA1c_VALUE1[t] > 6.5 and HBA1c_VALUE1[t] <= 7:  
        HBA1c_VALUE[t] = 1
        G1 = G1 + 1   
    elif HBA1c_VALUE1[t] > 7 and HBA1c_VALUE1[t] <= 7.5:
        HBA1c_VALUE[t] = 2
        G2 = G2 + 1  
    elif HBA1c_VALUE1[t] > 7.5 and HBA1c_VALUE1[t] <= 8:
        HBA1c_VALUE[t] = 3
        G3 = G3 + 1  
    elif HBA1c_VALUE1[t] > 8 and HBA1c_VALUE1[t] <= 8.5:
        HBA1c_VALUE[t] = 4
        G4 = G4 + 1 
    elif HBA1c_VALUE1[t] > 8.5 and HBA1c_VALUE1[t] <= 9:
        HBA1c_VALUE[t] = 5
        G5 = G5 + 1 
    elif HBA1c_VALUE1[t] > 9 and HBA1c_VALUE1[t] <= 10:
        HBA1c_VALUE[t] = 6
        G6 = G6 + 1 
    elif HBA1c_VALUE1[t] > 10 and HBA1c_VALUE1[t] <= 11:
        HBA1c_VALUE[t] = 7
        G7 = G7 + 1 
    elif HBA1c_VALUE1[t] > 11:
        HBA1c_VALUE[t] = 8
        G8 = G8 + 1 
        
print(G0,G1,G2,G3,G4,G5,G6,G7,G8)



BATCH_SIZE = 32

x1 = np.linspace(1,38400,38400)
x2 = np.linspace(1,38400,75000)
x3 = np.linspace(1,38400,150000)
train_img_list= []
train_label_list = []
test_img_list= []
test_label_list = []

for m in range(len(USE_ID_list)):            
    path_ID = str(USE_ID_list[m])
    img_list = os.listdir(path1 + '/' + path_ID)
    print(path_ID, len(img_list))
    if len(img_list) < 20 and len(img_list) > 0:
       for img in img_list:  
           label = 0
           all_data = scio.loadmat(path1 + '/' + path_ID + '/' + img)
           oriECG_data = all_data["ECG_data"]
           length = oriECG_data.size
           if length >= 38400 and length < 40000:
               new_ECG = oriECG_data[0,0:38400]
               label = 128
           elif length >= 75000 and length < 85000:
               new_ECG = oriECG_data[0,0:75000]
               label = 250
           elif length >= 150000 and length < 160000:
               new_ECG = oriECG_data[0,0:150000]
               label = 500
           
           if label == 128: 
               resize_ECG = np.interp(x2, x1, new_ECG)              
           elif label == 500: 
               resize_ECG = np.interp(x2, x3, new_ECG)               
           else:
               resize_ECG = new_ECG
               
           norm_data = np.zeros((5,15000))
           resize_ECG = resize_ECG.reshape(-1,15000)
           for i in range(5):
               norm = preprocessing.MinMaxScaler()
               norm_s = norm.fit_transform(resize_ECG[i].reshape(-1,1))
               norm_s = norm_s.reshape(1,15000)
               norm_data[i] = norm_s
           train_img_list.append(norm_data)
           train_HBA1c =  HBA1c_VALUE[m]
           train_label_list.append(train_HBA1c)
           train_label_list.append(train_HBA1c)
           train_label_list.append(train_HBA1c)
           train_label_list.append(train_HBA1c)
           train_label_list.append(train_HBA1c) 
           
    elif len(img_list) >= 20: 
        
        P1 = int(0.6*len(img_list))
        P2 = int(0.8*len(img_list))            
        Y = 0
        M = 0
        for img in img_list:
            M = M + 1
            Y = Y + 1
            if Y < 200:
                label = 0
                all_data = scio.loadmat(path1 + '/' + path_ID + '/' + img)
                oriECG_data = all_data["ECG_data"]
                length = oriECG_data.size
        
                if length >= 38400 and length < 40000:
                    new_ECG = oriECG_data[0,0:38400]
                    label = 128
                elif length >= 75000 and length < 85000:
                    new_ECG = oriECG_data[0,0:75000]
                    label = 256
                elif length >= 150000 and length < 160000:
                    new_ECG = oriECG_data[0,0:150000]
                    label = 500
        
                if label == 128:
                    resize_ECG = np.interp(x2, x1, new_ECG)
                elif label == 500:
                    resize_ECG = np.interp(x2, x3, new_ECG)
                else:
                    resize_ECG = new_ECG  
                    
                norm_data = np.zeros((5,15000))
                resize_ECG = resize_ECG.reshape(-1,15000)
                for i in range(5):
                    norm = preprocessing.MinMaxScaler()
                    norm_s = norm.fit_transform(resize_ECG[i].reshape(-1,1))
                    norm_s = norm_s.reshape(1,15000)
                    norm_data[i] = norm_s        
               
                if M >= P1 and M < P2:
                    test_img_list.append(norm_data)
                    test_HBA1c =  HBA1c_VALUE[m]
                    test_label_list.append(test_HBA1c)
                    test_label_list.append(test_HBA1c)
                    test_label_list.append(test_HBA1c)
                    test_label_list.append(test_HBA1c)
                    test_label_list.append(test_HBA1c)                     
                else:
                    train_img_list.append(norm_data)
                    train_HBA1c =  HBA1c_VALUE[m]
                    train_label_list.append(train_HBA1c)
                    train_label_list.append(train_HBA1c)
                    train_label_list.append(train_HBA1c)
                    train_label_list.append(train_HBA1c)
                    train_label_list.append(train_HBA1c)                    

train_img = np.array(train_img_list)
train_img = train_img.reshape(-1,15000)
train_img = np.expand_dims(train_img, axis=1)
train_label = np.array(train_label_list) 




test_img = np.array(test_img_list)
test_img = test_img.reshape(-1,15000)
test_img = np.expand_dims(test_img, axis=1)
test_label = np.array(test_label_list) 

del  train_img_list, test_img_list
gc.collect()  

train_img, train_label = shuffle(train_img, train_label, random_state = 128)         
           

train_img = torch.from_numpy(train_img)
train_label = torch.from_numpy(train_label)
test_img = torch.from_numpy(test_img)
test_label = torch.from_numpy(test_label)    

train_set = torch.utils.data.TensorDataset(train_img, train_label)
test_set = torch.utils.data.TensorDataset(test_img, test_label)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False) 

del  train_img, train_label, test_img, test_label, train_set ,test_set
gc.collect()  



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


cnn=cnn.to(DEVICE)
criterion = NN.CrossEntropyLoss().to(DEVICE)



LEARNING_RATE = 0.0001
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
TOTAL_EPOCHS = 460
losses = []
epoch_list = []
train_accu_list = []
test_accu_list = [] 
    
for epoch in range(TOTAL_EPOCHS):
    train_loss, test_losses = 0, 0
    train_correct = 0
    train_total = 0
    for p, (images, labels) in enumerate(train_loader):
        labels = labels.view(-1)
        labels = labels.numpy()
        repeat_train_label = np.repeat(labels, 20)
        repeat_train_label = torch.from_numpy(repeat_train_label)
        images = images.view(-1,1,750)
        images = images.float().to(DEVICE)
        repeat_train_label = repeat_train_label.to(DEVICE).long()

        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, repeat_train_label)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
        train_outputs = outputs.cpu()
        repeat_train_label = repeat_train_label.cpu()
        _, train_predicted = torch.max(train_outputs.data, 1)
        train_total += repeat_train_label.size(0)
        train_correct += (train_predicted == repeat_train_label).sum()
   

     
    cnn.eval() 
    test_correct = 0
    test_total = 0
    for q, (test_images, test_labels) in enumerate(test_loader):
        test_labels = test_labels.view(-1)
        test_images = test_images.view(-1,1,750)
        test_images = test_images.float().to(DEVICE)
        test_outputs = cnn(test_images).cpu()
        test_labels = test_labels.cpu()
        test_predicted = F.softmax(test_outputs, dim=1)
        test_predicted = test_predicted.detach().numpy()
        test_predicted = test_predicted.reshape(-1,20,9)
        test_len = len(test_predicted)
        new_test_list = []
        for t in range (test_len):
            temp_label = test_predicted[t,:]
            label_max = temp_label.max()
            label_min = temp_label.min()
            R = 20
            C = 9
            sign_label = 0
            total_label = 0
            for m in range (C):
                ad_value_all = 0
                for n in range(R):
                    min_value = (temp_label[n,:]).min()
                    ad_value = (temp_label[n,m] - min_value) / (label_max - label_min) * temp_label[n,m]
                    ad_value_all = ad_value_all + ad_value
                if  ad_value_all >  total_label:
                    total_label =  ad_value_all
                    sign_label = m
                        
            new_test_list.append(sign_label)
        new_test_label = np.array(new_test_list)
        true_test_label = test_labels.numpy()
        for t in range (test_len):
            if new_test_label[t] == true_test_label[t]:
               test_correct = test_correct + 1
        test_total = test_total + test_len
          
        
    torch.save(cnn, path3 + "/" + "CNN_MFVW_EPOCHS " + str(epoch+1) + ".pth")     
    print('Epoch: %d, train_acc: %.4f, test_acc: %.4f' %(epoch + 1, (train_correct.numpy()) / train_total, test_correct / test_total)) 
    epoch_list.append(epoch + 1)
    train_accu_list.append((train_correct.numpy()) / train_total)
    test_accu_list.append(test_correct / test_total)    
epoch_num = np.array(epoch_list)  
train_accu = np.array(train_accu_list)
test_accu = np.array(test_accu_list)
result = np.concatenate((epoch_num.reshape(-1,1),train_accu.reshape(-1,1),test_accu.reshape(-1,1)),axis=1)
np.savetxt(path3 + "/" + "CNN_MFVW_accuracy" + ".csv", result, fmt='%.4f', delimiter = ',')