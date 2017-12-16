import numpy as np
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
#from tensorflow.examples.tutorials.mnist import input_data
from torch.autograd import Variable
import sklearn.metrics as metrics
import pickle
import os
import numpy.linalg as la
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sys
import pandas as pd



def get_features(f_index=0,b=True,c=True,train=True,cleaned = False):
    
    import pickle
    import os
    
    with open('screen_info.txt','rb') as fl:
        t = pickle.load(fl)
    fnames = t[0]
    totf = t[1]
    binf = t[2]

    fname = fnames[f_index]
    bf = binf[f_index]
    path = os.getcwd() + '/bioassay-datasets/'
    p_fingerprints = []
    c_fingerprints = []
    labels = []
    if train is True:
        if cleaned is False:
            with open(path+fname+'red_train.csv') as csvfile:
                readcsv = csv.reader(csvfile)
                for row in readcsv:
                    p_fingerprints.append(row[:bf])
                    c_fingerprints.append(row[bf:-1])
                    labels.append(row[-1])
        else:
            path = os.getcwd() + '/bioassay-datasets/cleaned_'+str(f_index)+'.csv'
            with open(path) as csvfile:
                readcsv = csv.reader(csvfile)
                for row in readcsv:
                    p_fingerprints.append(row[1:bf+1])
                    c_fingerprints.append(row[bf+1:-1])
                    labels.append(row[-1])
    else:
        with open(path+fname+'red_test.csv') as csvfile:
            readcsv = csv.reader(csvfile)
            for row in readcsv:
                p_fingerprints.append(row[:bf])
                c_fingerprints.append(row[bf:-1])
                labels.append(row[-1])
        
    p_fingerprints = np.asarray(p_fingerprints)[1:]
    p_fingerprints = p_fingerprints.astype(float)

    c_fingerprints = np.asarray(c_fingerprints)[1:]
    c_fingerprints = c_fingerprints.astype(float)

    labels = labels[1:]
    #Normalise the features
    c_fingerprints = (c_fingerprints - np.mean(c_fingerprints,axis=0))/np.std(c_fingerprints,axis=0)

    fingerprints = np.concatenate((p_fingerprints,c_fingerprints),axis=1)

    labels2 = np.zeros((len(labels),1))
    
    for i,l in enumerate(labels):
        if l=='Active':
            labels2[i] = 1
        else:
            labels2[i] = 0
    labels2 = labels2.astype(int)
    
    if cleaned is True:
        labels2 = np.asarray(labels)
        labels2 = labels2.astype(float)
    if(b is True and c is True):
        return fingerprints,labels2
    elif b is True:
        return p_fignerprints,labels2
    else:
        return c_fingerprints,labels2
    
def shuffle(x,y):
  
    prev_shape_x = x.shape
    prev_shape_y = y.shape
    ind = np.arange(prev_shape_x[0])
    np.random.shuffle(ind)
    x = x[ind].reshape(prev_shape_x)
    y = y[ind].reshape(prev_shape_y)
    return x,y

def get_train_ind(val_iter,no_examples):
    
    if val_iter == 0: #no validation
        curr_data_size = no_examples
    else:
        curr_data_size = int(no_examples*0.8)
        interval_size = int(no_examples*0.2)
        
        if(val_iter==1):
            s_ind1 = int((val_iter)*interval_size)
            end_ind1 = int((val_iter+1)*interval_size)
            s_ind2 = int((val_iter + 1) * interval_size)
            end_ind2 = int(no_examples)
        else:
            s_ind1 = 0
            end_ind1 = int((val_iter-1)*interval_size)
            s_ind2 = int((val_iter) * interval_size)
            end_ind2 = int(no_examples)
        
        #print("train_ind ",s_ind1,end_ind1,s_ind2,end_ind2)
        indices = range(s_ind1,end_ind1) + range(s_ind2,end_ind2)
        return indices
        #for debugging
        #return s_ind1,end_ind1,s_ind2,end_ind2 
    
def get_train_batch(x,y,indices,batch_size,cuda=True):
    
    train_data = x[indices]
    labels_train = y[indices]
    curr_data_size = train_data.shape[0]
    
    samples = np.random.randint(low=0,high=curr_data_size,size=(batch_size,1))
        
    train_batch = train_data[samples].reshape(batch_size,x.shape[1])
    train_batch = train_batch.astype(float)
    
    if cuda is True:
        train_batch = torch.cuda.FloatTensor(train_batch)
        train_batch = Variable(train_batch,requires_grad=False).cuda()
        labels_train = Variable(torch.cuda.LongTensor(labels_train[samples]),requires_grad=False)
        labels_train = labels_train.view(batch_size,)
    return train_batch,labels_train

def get_val_data(x,y,no_examples,val_iter,cuda=True):
    
    interval_size = int(no_examples)*0.2
    s_ind = int((val_iter-1)*interval_size)
    e_ind = int((val_iter) * interval_size)
    
    train_data = x[s_ind:e_ind]
    labels_val = y[s_ind:e_ind] 
   
    
    if cuda is True:
        return Variable(torch.cuda.FloatTensor(train_data)),labels_val  
    else:
        return train_data,labels_val
    #for debugging
    #return s_ind,e_ind