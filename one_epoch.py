#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
tf.config.list_physical_devices('GPU')

from tensorflow import keras
from keras.models import model_from_json, load_model
from keras.callbacks import ModelCheckpoint

import Data
import DataBatcher
import warnings
warnings.filterwarnings('ignore')


# In[5]:

class One_epoch:
    def __init__(self,batch_size=2048,model='CNN'):
        self.batch_size=batch_size

        self.low_ele=Data.LoadData('Dataset_information/electrons_100GeV_1TeV_path_size')
        self.high_ele=Data.LoadData('Dataset_information/electrons_1TeV_20TeV_path_size')
        self.low_pro=Data.LoadData('Dataset_information/protons_100GeV_1TeV_path_size')
        self.high_pro=Data.LoadData('Dataset_information/protons_1TeV_10TeV_path_size')
    
        self.low_ele_path_tree=self.low_ele.training_path_tree(0.5)
        self.high_ele_path_tree=self.high_ele.training_path_tree(0.5)
        self.low_pro_path_tree=self.low_pro.training_path_tree(0.5)
        self.high_pro_path_tree=self.high_pro.training_path_tree(0.5)

        self.low_ele_size=self.low_ele.training_size(0.5)
        self.high_ele_size=self.high_ele.training_size(0.5)
        self.low_pro_size=self.low_pro.training_size(0.5)
        self.high_pro_size=self.high_pro.training_size(0.5)
        
        self.massimo=self.low_ele.massimo
        self.model=model

# In[6]:
    def train_one_epoch(self,epoch_id,l_r):
        checkpoint = ModelCheckpoint(self.model+'_model_trained/'+self.model+'-'+str(self.batch_size)+'-'+str(epoch_id)+'.h5', verbose=1,save_best_only=False, mode='auto')
        model = load_model(self.model+'_model_trained/'+self.model+'-'+str(self.batch_size)+'-'+str(epoch_id-1)+'.h5')

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=l_r),loss='binary_crossentropy',metrics=['accuracy'])
        object=DataBatcher.BatcherCreation(self.low_ele_path_tree,self.high_ele_path_tree,self.low_pro_path_tree,
                                           self.high_pro_path_tree,self.low_ele_size,self.high_ele_size,self.low_pro_size,
                                           self.high_pro_size,self.massimo,self.batch_size)

        history=model.fit(object,epochs=1,shuffle=True,callbacks=[checkpoint],verbose=1)
        with open(self.model+'_model_trained/training_log.txt','a') as f:
            f.write(str(epoch_id)+'   '+str(l_r)+'   :'+str(history.history['loss'])+'   '+str(history.history['accuracy'])+'\n')