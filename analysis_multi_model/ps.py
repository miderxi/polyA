import pandas as pd
import numpy  as np

def get_onehot_data(poly_name,file_dir='../data/'):
    '''
    load and preprocesing data into one-hot encoding
    加载数据并将数据转化成独热码
    
    '''
    base_one_hot={
    'A':np.array([1,0,0,0],dtype='float16'),
    'T':np.array([0,1,0,0],dtype='float16'),
    'C':np.array([0,0,1,0],dtype='float16'),
    'G':np.array([0,0,0,1],dtype='float16'),
    'a':np.array([1,0,0,0],dtype='float16'),
    't':np.array([0,1,0,0],dtype='float16'),
    'c':np.array([0,0,1,0],dtype='float16'),
    'g':np.array([0,0,0,1],dtype='float16')
    }
    
    file_path_pos = file_dir+poly_name+'.txt'
    file_path_neg = file_dir+'neg'+poly_name+'.txt'

    pdata = np.loadtxt(file_path_pos,dtype='str')
    pdata = [seq[:100]+seq[106:] for seq in pdata]#remove the polya signal
    pdata = [[base_one_hot[base] for base in seq] for seq in pdata]
                   
    ndata = np.loadtxt(file_path_neg,dtype='str')
    ndata = [seq[:100]+seq[106:] for seq in ndata]#remove the polya signal
    ndata = [[base_one_hot[base] for base in seq] for seq in ndata]
   
    X = np.array(pdata+ndata)#pdata,ndata->list; #X(-1,200,4)
    y = np.array(np.append( np.ones(len(pdata)),np.zeros(len(ndata)) ))
    return X,y


# from keras.models import Sequential
# from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout,Activation
# from keras.optimizers        import Adam    
# def cnn_model(input_shape = (1,200,4)):
#     model = Sequential()
#     model.add(Conv2D(filters=64,kernel_size=(6,4),
#                     padding='same',activation='relu',
#                     input_shape=input_shape))
#     model.add(Conv2D(filters=64,kernel_size=(8,4),
#                     padding='same',activation='relu'))
#     model.add(MaxPool2D(pool_size=(2,2),padding='same'))
#     model.add(Flatten())
#     model.add(Dense(128,activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(1,activation='sigmoid'))
#     adam=Adam(lr=0.5*1e-4)
#     model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])    
#     return model
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv1D,MaxPool1D,Flatten,Dense,Dropout,Activation
from keras.optimizers        import Adam  
def cnn_model(input_shape=(200,4)):
    model = Sequential()
    model.add(Conv1D(64,7,padding='same',activation='relu',input_shape=input_shape,kernel_regularizer=regularizers.l1_l2(0.001,0.001)))
    model.add(MaxPool1D(2))
    
    model.add(Conv1D(64,7,padding='same',activation='relu',kernel_regularizer=regularizers.l1_l2(0.001,0.001)))
    model.add(MaxPool1D(2))
    
    model.add(Conv1D(128,7,padding='same',activation='relu',kernel_regularizer=regularizers.l1_l2(0.001,0.001)))
    model.add(MaxPool1D(2))
    
    model.add(Conv1D(128,7,padding='same',activation='relu'))
    model.add(MaxPool1D(2))
    
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))
    adam=Adam(lr=1*1e-3)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])    
    return model    

def small_model():
    model = Sequential()
    model.add(Conv1D(64,7,padding='same',activation='relu',input_shape=(200,4),kernel_regularizer=regularizers.l1_l2(0.001,0.001)))
    model.add(MaxPool1D(2))
    model.add(Conv1D(64,7,padding='same',activation='relu'))
    model.add(MaxPool1D(2))
    
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    return model   

from sklearn import metrics
import matplotlib.pyplot as plt
def assess(y_true,y_prob,roc=True,pr=False,poly_name=None,line_name=None):
    '''
    pass
    '''
    re={}
    y_true = np.array(y_true,dtype=int)
    y_pred = np.array(y_prob+0.5,dtype=int)
    
    re['accuracy'] = round(metrics.accuracy_score  (y_true,y_pred),3)
    re['precision'] = round(metrics.precision_score(y_true,y_pred),3)
    re['recall'] = round(metrics.recall_score      (y_true,y_pred),3)
    re['f1'] =  round(metrics.f1_score             (y_true,y_pred),3)
    re['auc'] = round(metrics.roc_auc_score        (y_true, y_prob),3)
    
    if (poly_name != None):
        my_label = poly_name+': '+str(re['auc'])
    elif(line_name != None):
        my_label = line_name+': '+str(re['auc'])
    else:
        my_label = 'roc'+': '+str(re['auc'])
    
    fpr,tpr,thresholds = metrics.roc_curve(y_true,y_prob)
    plt.plot(fpr,tpr,label=my_label)
    plt.title('ROC curve')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    
    #precision, recall, thresholds = metrics.precision_recall_curve(y_true,y_prob)
    #plt.plot(precision,recall,label='precison_and_recall')
    plt.legend()
    
    return re

import matplotlib.pyplot as plt
def showhist(history):
    fig,ax = plt.subplots(1,2,figsize=(8,2))
    ax[0].plot(history.history['accuracy'],label='train_acc')
    ax[0].plot(history.history['val_accuracy'],label='val_acc')
    ax[0].set_title('train and validation accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend( loc='lower right')    
    
    ax[1].plot(history.history['loss'],label='train_loss')
    ax[1].plot(history.history['val_loss'],label='val_loss')   
    ax[1].set_title('train and vaidation loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend( loc='upper right')    
    plt.show()
    


