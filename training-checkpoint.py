#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
tf.config.list_physical_devices('GPU')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers, models
from keras.models import Model,model_from_json
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import uproot as up
import pickle
import random
import gc

##### creiamo il DataLoader #####

# In[4]:

class Batcher():
    def __init__(self,size_array,names_array,batch_size=32):
        
        self.size_array=size_array
        self.names_array=names_array
        self.batch_size=batch_size



    def batches_creation(self):
        ev_nel_batch=0
        num_batch=0
        num_file=0
        ev_da_file=0
        ev_to_skip=0
        final_batch_origin=[[]]
        ev_nel_file=self.size_array[num_file]
        for i in range(sum(self.size_array)):
            ev_nel_batch+=1
            ev_nel_file-=1
            ev_da_file+=1
            if ev_nel_batch==self.batch_size:
                final_batch_origin[num_batch].append([self.names_array[num_file],ev_to_skip,ev_to_skip+ev_da_file])
                ev_to_skip+=ev_da_file
                ev_da_file=0
                num_batch+=1
                ev_nel_batch=0
                final_batch_origin.append([])
            if ev_nel_file==0:
                final_batch_origin[num_batch].append([self.names_array[num_file],ev_to_skip,ev_to_skip+ev_da_file])
                num_file+=1
                if num_file==len(self.size_array):
                    ev_nel_batch=0
                    num_batch=0
                    num_file=0
                    ev_da_file=0
                    ev_to_skip=0
                    break
                ev_nel_file=self.size_array[num_file]
                ev_to_skip=0
                ev_da_file=0
        return final_batch_origin


# In[5]:


class DataLoader(Batcher,Sequence):
    def __init__(self,
                 list_low_elcectrons,list_high_electrons,list_low_protons,list_high_protons,
                 list_low_electrons_size,list_high_electrons_size,list_low_protons_size,list_high_protons_size,
                 massimo,batch_size=64):
        
        self.low_electrons=list_low_elcectrons
        self.high_electrons=list_high_electrons
        self.low_protons=list_low_protons
        self.high_protons=list_high_protons

        self.low_electrons_size=list_low_electrons_size
        self.high_electrons_size=list_high_electrons_size
        self.low_protons_size=list_low_protons_size
        self.high_protons_size=list_high_protons_size

        self.massimo=massimo
        self.batch_size=batch_size

        self.low_ele_labels=[]
        self.high_ele_labels=[]
        self.low_pro_labels=[]
        self.high_pro_labels=[]

        self.dataset=pd.DataFrame()
        self.labels=[[]]

        self.perc_low_electrons=sum(self.low_electrons_size)/(sum(self.low_electrons_size)+sum(self.high_electrons_size)+sum(self.low_protons_size)+sum(self.high_protons_size))
        self.perc_high_electrons=sum(self.high_electrons_size)/(sum(self.low_electrons_size)+sum(self.high_electrons_size)+sum(self.low_protons_size)+sum(self.high_protons_size))
        self.perc_high_protons=sum(self.high_protons_size)/(sum(self.low_electrons_size)+sum(self.high_electrons_size)+sum(self.low_protons_size)+sum(self.high_protons_size))
        self.perc_low_protons=sum(self.low_protons_size)/(sum(self.low_electrons_size)+sum(self.high_electrons_size)+sum(self.low_protons_size)+sum(self.high_protons_size))

        self.low_electrons_per_batch=int(self.batch_size*self.perc_low_electrons)
        self.high_electrons_per_batch=int(self.batch_size*self.perc_high_electrons)
        self.low_protons_per_batch=int(self.batch_size*self.perc_low_protons)
        self.high_protons_per_batch=self.batch_size-self.low_electrons_per_batch-self.high_electrons_per_batch-self.low_protons_per_batch
        self.low_electrons_batch_per_file=Batcher(self.low_electrons_size,self.low_electrons,self.low_electrons_per_batch).batches_creation()
        self.high_electrons_batch_per_file=Batcher(self.high_electrons_size,self.high_electrons,self.high_electrons_per_batch).batches_creation()
        self.low_protons_batch_per_file=Batcher(self.low_protons_size,self.low_protons,self.low_protons_per_batch).batches_creation()
        self.high_protons_batch_per_file=Batcher(self.high_protons_size,self.high_protons,self.high_protons_per_batch).batches_creation()
           
        self.min_len=min(len(self.low_electrons_batch_per_file),len(self.high_electrons_batch_per_file),len(self.low_protons_batch_per_file),len(self.high_protons_batch_per_file))-1

        self.low_electrons_batch_per_file=self.low_electrons_batch_per_file[:self.min_len]
        self.high_electrons_batch_per_file=self.high_electrons_batch_per_file[:self.min_len]
        self.low_protons_batch_per_file=self.low_protons_batch_per_file[:self.min_len]
        self.high_protons_batch_per_file=self.high_protons_batch_per_file[:self.min_len]
        
        low_ele_ev_last_cell=0
        for low_ele_last_cell in self.low_electrons_batch_per_file[-1]:
            low_ele_ev_last_cell+=low_ele_last_cell[2]-low_ele_last_cell[1]
        
        high_ele_ev_last_cell=0
        for high_ele_last_cell in self.high_electrons_batch_per_file[-1]:
            high_ele_ev_last_cell+=high_ele_last_cell[2]-high_ele_last_cell[1]
        
        low_pro_ev_last_cell=0
        for low_pro_last_cell in self.low_protons_batch_per_file[-1]:
            low_pro_ev_last_cell+=low_pro_last_cell[2]-low_pro_last_cell[1]
        
        high_pro_ev_last_cell=0
        for high_pro_last_cell in self.high_protons_batch_per_file[-1]:
            high_pro_ev_last_cell+=high_pro_last_cell[2]-high_pro_last_cell[1]
        
        
        for i in range(self.min_len):
            self.low_ele_labels=([1]*self.low_electrons_per_batch)
            self.high_ele_labels=([1]*self.high_electrons_per_batch)
            self.low_pro_labels=([0]*self.low_protons_per_batch)
            self.high_pro_labels=([0]*self.high_protons_per_batch)
            self.labels[i]=(self.low_ele_labels+self.high_ele_labels+self.low_pro_labels+self.high_pro_labels)
            if i==self.min_len-1:
                break
            else:
                self.labels.append([])
        
    def __len__(self):
        return self.min_len
    
    def creation_dataset(self,idx):
        matr_batch=[]
        for i in range(int(self.dataset.shape[0]/400)):
            matrix = np.array(self.dataset.iloc[i*400:(i+1)*400]).reshape(20, 20)
            matrix=matrix/self.massimo
            matr_batch.append(matrix)

        shuffled=list(zip(matr_batch,self.labels[idx]))
        random.shuffle(shuffled)

        self.dataset=pd.DataFrame()

        inputs,targets=zip(*shuffled)
        inputs=np.array(inputs)
        targets=np.array(targets)

        return inputs,targets
    
    def __getitem__(self,idx):
        for low_ele_events in self.low_electrons_batch_per_file[idx]:
            with up.open(low_ele_events[0][:-14]) as file:
                low_ele_df = file['showersTree'].arrays('deps2D', library='pd')[low_ele_events[1]*400:low_ele_events[2]*400]
                self.dataset=pd.concat([self.dataset,low_ele_df])
        
        for high_ele_events in self.high_electrons_batch_per_file[idx]:
            with up.open(high_ele_events[0][:-14]) as file:
                high_ele_df = file['showersTree'].arrays('deps2D', library='pd')[high_ele_events[1]*400:high_ele_events[2]*400]
                self.dataset=pd.concat([self.dataset,high_ele_df])
        
        for low_pro_events in self.low_protons_batch_per_file[idx]:
            with up.open(low_pro_events[0][:-14]) as file:
                low_pro_df = file['showersTree'].arrays('deps2D', library='pd')[low_pro_events[1]*400:low_pro_events[2]*400]
                self.dataset=pd.concat([self.dataset,low_pro_df])
            
        for high_pro_events in self.high_protons_batch_per_file[idx]:
            with up.open(high_pro_events[0][:-14]) as file:
                high_pro_df = file['showersTree'].arrays('deps2D', library='pd')[high_pro_events[1]*400:high_pro_events[2]*400]
                self.dataset=pd.concat([self.dataset,high_pro_df])

        return self.creation_dataset(idx)

##### prepariamo i dati per il training ####
# In[6]:


with open('Dataset_information/electrons_100GeV_1TeV_path_size','rb') as f:
    electrons_100GeV_1TeV_path_size=pickle.load(f)
    
with open('Dataset_information/electrons_1TeV_20TeV_path_size','rb') as f:
    electrons_1TeV_20TeV_path_size=pickle.load(f)

with open('Dataset_information/protons_100GeV_1TeV_path_size','rb') as f:
    protons_100GeV_1TeV_path_size=pickle.load(f)

with open('Dataset_information/protons_1TeV_10TeV_path_size','rb') as f:
    protons_1TeV_10TeV_path_size=pickle.load(f)

electrons_100GeV_1TeV_path=[row[0] for row in electrons_100GeV_1TeV_path_size]
electrons_1TeV_20TeV_path=[row[0] for row in electrons_1TeV_20TeV_path_size]

protons_100GeV_1TeV_path=[row[0] for row in protons_100GeV_1TeV_path_size]
protons_1TeV_10TeV_path=[row[0] for row in protons_1TeV_10TeV_path_size]

electrons_100GeV_1TeV_size=[row[1] for row in electrons_100GeV_1TeV_path_size]
electrons_1TeV_20TeV_size=[row[1] for row in electrons_1TeV_20TeV_path_size]

protons_100GeV_1TeV_size=[row[1] for row in protons_100GeV_1TeV_path_size]
protons_1TeV_10TeV_size=[row[1] for row in protons_1TeV_10TeV_path_size]

electrons_100GeV_1TeV_training_size=sum(electrons_100GeV_1TeV_size)/2
electrons_1TeV_20TeV_training_size=sum(electrons_1TeV_20TeV_size)/2

protons_100GeV_1TeV_training_size=sum(protons_100GeV_1TeV_size)/2
protons_1TeV_10TeV_training_size=sum(protons_1TeV_10TeV_size)/2


# In[7]:


somma=0
for i in range(len(electrons_100GeV_1TeV_size)):
    somma+=electrons_100GeV_1TeV_size[i]
    if somma > electrons_100GeV_1TeV_training_size:
        final_electrons_100GeV_1TeV_training_file=i+1
        break

somma=0
for i in range(len(electrons_1TeV_20TeV_size)):
    somma+=electrons_1TeV_20TeV_size[i]
    if somma > electrons_1TeV_20TeV_training_size:
        final_electrons_1TeV_20TeV_training_file=i+1
        break


somma=0
for i in range(len(protons_100GeV_1TeV_size)):
    somma+=protons_100GeV_1TeV_size[i]
    if somma > protons_100GeV_1TeV_training_size:
        final_protons_100GeV_1TeV_training_file=i+1
        break

somma=0
for i in range(len(protons_1TeV_10TeV_size)):
    somma+=protons_1TeV_10TeV_size[i]
    if somma > protons_1TeV_10TeV_training_size:
        final_protons_1TeV_10TeV_training_file=i+1
        break


# In[8]:


electrons_100GeV_1TeV_path_tree=[name+':showersTree;1' for name in electrons_100GeV_1TeV_path]
electrons_1TeV_20TeV_path_tree=[name+':showersTree;1' for name in electrons_1TeV_20TeV_path]

protons_100GeV_1TeV_path_tree=[name+':showersTree;1' for name in protons_100GeV_1TeV_path]
protons_1TeV_10TeV_path_tree=[name+':showersTree;1' for name in protons_1TeV_10TeV_path]


# In[9]:


with open('Dataset_information/max_training_values','rb') as f:
    massimo=pickle.load(f)

with open('Dataset_information/min_training_values','rb') as f:
    minimo=pickle.load(f)


# In[10]:


training_set_path_tree=electrons_100GeV_1TeV_path_tree[:final_electrons_100GeV_1TeV_training_file]+electrons_1TeV_20TeV_path_tree[:final_electrons_1TeV_20TeV_training_file]+protons_100GeV_1TeV_path_tree[:final_protons_100GeV_1TeV_training_file]+protons_1TeV_10TeV_path_tree[:final_protons_1TeV_10TeV_training_file]

list_electrons_training_size=electrons_100GeV_1TeV_size[:final_electrons_100GeV_1TeV_training_file]+electrons_1TeV_20TeV_size[:final_electrons_1TeV_20TeV_training_file]

list_protons_training_size=protons_100GeV_1TeV_size[:final_protons_100GeV_1TeV_training_file]+protons_1TeV_10TeV_size[:final_protons_1TeV_10TeV_training_file]

list_training_size=list_electrons_training_size+list_protons_training_size

#### prepariamo la schedule del training ####
# In[11]:

batch_size=128


def scheduler(epoch, lr):
    decay_rate = 0.94
    if epoch % 2 == 0 and epoch != 0:
        return lr * decay_rate
    else:
        return lr

callback = keras.callbacks.LearningRateScheduler(scheduler)

checkpoint = ModelCheckpoint('model_trained/CNN-{epoch:03d}-'+str(batch_size)+'.h5', verbose=1, save_best_only=False, mode='auto')


##### cariachiamo il modello #####

# In[12]:


with open('model_pre-trained/CNN.json', 'r') as f:
    modello_json = f.read()    
model= model_from_json(modello_json)
model.compile(optimizer=keras.optimizers.SGD(lr=0.045, momentum=0.9), loss='sparse_categorical_crossentropy',metrics=['accuracy'])


##### creaimo l'oggetto DataLoader ####
# In[13]:

object=DataLoader(electrons_100GeV_1TeV_path_tree[:final_electrons_100GeV_1TeV_training_file],electrons_1TeV_20TeV_path_tree[:final_electrons_1TeV_20TeV_training_file]
                   ,protons_100GeV_1TeV_path_tree[:final_protons_100GeV_1TeV_training_file],protons_1TeV_10TeV_path_tree[:final_protons_1TeV_10TeV_training_file],
                   electrons_100GeV_1TeV_size[:final_electrons_100GeV_1TeV_training_file],electrons_1TeV_20TeV_size[:final_electrons_1TeV_20TeV_training_file],
                   protons_100GeV_1TeV_size[:final_protons_100GeV_1TeV_training_file],protons_1TeV_10TeV_size[:final_protons_1TeV_10TeV_training_file],massimo,batch_size)


#### trainiamo il modello ####

# In[ ]:


model.fit(object,epochs=50,shuffle=True,callbacks=[callback,checkpoint],verbose=1)




