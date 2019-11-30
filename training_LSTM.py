#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:59:09 2019

@author: rcuello
"""
import pickle
import torch
import numpy as np
import time
#import torchvision.transforms as transforms
import pandas as pd
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Compose, Resize, ToTensor, CenterCrop, Normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt

if_gpu = torch.cuda.is_available()
print("GPU is on?", if_gpu)

def load_dataset(file_name="MainDataSet.pickle"):
    with open(file_name,'rb') as f:
        data = pickle.load(f)  
    return data

class DataRepo():
    def __init__(self,header,data):
        self.header = header
        self.data = data['LSTM']
        fcl = data['ClassList']
        fcl['ClassNames'] = fcl.apply(lambda row: row.ExerciseType+'-'+row.SampleType,axis=1)
        self.ClassList = fcl['ClassNames'].values.tolist()        
    def __len__(self):
        return(len(self.header))
    
    def getData(self,idx):
        ix,label = self.header.iloc[idx]
        img = self.data[ix]
        return img, label
    
    def getClassNames(self):
        return self.ClassList   
    
class GymnDataset(Dataset):
    
    def __init__(self,data_repo,seq_dim, input_dim,transform=None):
        self.data_repo = data_repo
        self.transform = transform
        self.seq_dim = seq_dim
        self.input_dim = input_dim
        
    def __len__(self):
        return(len(self.data_repo))
        
    def __getitem__(self,idx):
        img, label = self.data_repo.getData(idx)
        #img = np.array(img).trnaspose()
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        img = img.view(self.seq_dim,self.input_dim)
        label = torch.tensor(label)
        return img, label
    
    def getClassNames(self):
        return self.data_repo.getClassNames() 
    
    
def getBalancedList(data,test_size=0.15):
    imgs = list(range(len(data['CNN'])))
    mydict = {'SceneIndex':imgs,'Label':data['Labels']}
    odf = pd.DataFrame(mydict)
    y = odf['Label']
    return train_test_split(odf,test_size=test_size,stratify=y)

def getDataLoaders(data,batch_size,transforms,test_ratio=0.15):
    X_train, X_test = getBalancedList(data,test_ratio)
    train_repo = DataRepo(X_train,data)
    test_repo = DataRepo(X_test,data)
    trainData = GymnDataset(train_repo,510,4,transforms)
    testData = GymnDataset(test_repo,510,4,transforms)
    trainLoader = DataLoader(trainData,batch_size=batch_size)
    testLoader = DataLoader(testData,batch_size=batch_size)
    return trainLoader, testLoader    

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
    def loss(self,prediction,true_values):
        print(prediction.shape,'-',true_values.shape)
        print('* ',true_values)
        print('# ',prediction)
        return nn.CrossEntropyLoss(prediction, true_values)
        
def train(model, device, train_dataloader, optimizer, epoch, verbose=False):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (image, label) in enumerate(train_dataloader):
        input_var = image.to(device).float()
        print('Size=',input_var.shape)
        target_var = label.to(device)
        optimizer.zero_grad()
        output = model(input_var)
        #loss = model.loss(output, target_var)
        loss = criterion(output,target_var)
        #print(output)
        loss.backward()
        optimizer.step()
        total_loss += loss
        if batch_idx % 10 == 0 and verbose:
            print('Train Epoch: {0}, Train batch: {1}, Batch loss: {2}'
                  .format(epoch, batch_idx, loss))
    
    epoch_loss = total_loss/(batch_idx + 1)
    return epoch_loss        
        
def EvalModel(model, device, fortesting_dataloader):
    model.eval()
    evalStats = {}
    correct = 0
    total = 0
    preds = np.empty((0,1), int)
    with torch.no_grad():
        for image, label in fortesting_dataloader:
            input_var = image.to(device).float()
            target_var = label.to(device)
            output = model(input_var)
            _, predicted = torch.max(output.data, 1)
            #print(output.data)
            total += target_var.size(0)
            correct += (predicted == target_var).sum()
#            print(preds.shape,' - ',predicted.cpu().numpy().reshape(8,1))
 #           preds = np.append(preds,predicted.cpu().numpy(),axis=0)
            
    evalStats['Accuracy'] = correct / total
    evalStats['Predictions'] = preds.astype(int)
    return evalStats

def test(model, device, test_dataloader):
    eStats = EvalModel(model, device, test_dataloader)
    return eStats['Accuracy']

def getClassificationData(model,device,a_dataloader):
    cData = {}
    eStats = EvalModel(model, device, a_dataloader)
    df = a_dataloader.dataset.df.copy()
    labelList = a_dataloader.dataset.getClassNames()
    df['Class'] = df.Label.values.astype(int)
    df['PredictedClass'] = eStats['Predictions']
    cData['ClassNames'] = labelList
    cData['Data'] = df
    cData['CM'] = pd.DataFrame(confusion_matrix(df.Class.values, df.PredictedClass.values),
             columns=["P_" + class_name for class_name in labelList],
             index = [class_name for class_name in labelList])
    return cData

def printClassificationReport(cData):
    print(classification_report(cData['Data'].Class.values, cData['Data'].PredictedClass.values,target_names=cData['ClassNames']))

    
# main

losses = []
accuracies = []
n_epochs = 100
learning_rate = 0.1
lr_decay = 5 # every 10 epochs, the learning rate is divided by 10
batch_size = 8
use_cuda = False # use True to switch to GPU
mydata = load_dataset()

input_dim = 4
hidden_dim = 1020
layer_dim = 1
output_dim = 11
seq_dim = 510

device = torch.device("cuda" if use_cuda else "cpu")
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    
print('The model has {0} parameters'.format(sum([len(i.reshape(-1)) for i in model.parameters()]) ))

data_transformers = Compose([ToTensor()])
train_loader, test_loader = getDataLoaders(mydata, batch_size, data_transformers, test_ratio=0.1)
print('Number of train examples: {0}, number of test examles: {1}'
      .format(len(train_loader.dataset), len(test_loader.dataset)))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.96)
best_acc = 0

for epoch in range(1, n_epochs + 1):
    tic = time.time()
    epoch_loss = train(model, device, train_loader, optimizer, epoch)
    print('Training loss for epoch {0} is {1:.5f}.   LR={2}'.format(epoch, epoch_loss,lr_scheduler.get_lr()))
    losses.append(epoch_loss)
    accuracy = test(model, device, test_loader)
    print('Test accuracy: {0:.3f}'.format(accuracy))
    accuracies.append(accuracy)
    if accuracy>best_acc:
        best_acc = accuracy
        torch.save(model,'LSTM.pth')
        print('Best model saved')
    lr_scheduler.step()
    tac = time.time()
    print('Epoch time: {0:0.1f} seconds'.format(tac - tic))    