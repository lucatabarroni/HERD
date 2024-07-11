import tensorflow as tf
tf.config.list_physical_devices('GPU')

from tensorflow import keras
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

import Data
import DataBatcher
import warnings
warnings.filterwarnings('ignore')

batch_size=int(input('Scegli un batch size: '))

low_ele=Data.LoadData('Dataset_information/electrons_100GeV_1TeV_path_size')
high_ele=Data.LoadData('Dataset_information/electrons_1TeV_20TeV_path_size')
low_pro=Data.LoadData('Dataset_information/protons_100GeV_1TeV_path_size')
high_pro=Data.LoadData('Dataset_information/protons_1TeV_10TeV_path_size')

low_ele_path_tree=low_ele.training_path_tree(0.5)
high_ele_path_tree=high_ele.training_path_tree(0.5)
low_pro_path_tree=low_pro.training_path_tree(0.5)
high_pro_path_tree=high_pro.training_path_tree(0.5)

low_ele_size=low_ele.training_size(0.5)
high_ele_size=high_ele.training_size(0.5)
low_pro_size=low_pro.training_size(0.5)
high_pro_size=high_pro.training_size(0.5)

massimo=low_ele.massimo

model_name=input("Inserire il modello da trainare (CNN/ResNet_CNN) :")

with open(model_name+'_model_pre-trained/'+model_name+'.json', 'r') as f:
    modello_json = f.read() 
model= model_from_json(modello_json)
model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0010), loss='binary_crossentropy',metrics=['accuracy'])

checkpoint = ModelCheckpoint(model_name+'_model_trained/'+model_name+'-'+str(batch_size)+'-0.h5', verbose=1,save_best_only=False, mode='auto')

object=DataBatcher.BatcherCreation(low_ele_path_tree,high_ele_path_tree,low_pro_path_tree,high_pro_path_tree,low_ele_size,high_ele_size,low_pro_size,high_pro_size,massimo,batch_size)

history=model.fit(object,epochs=1,shuffle=False,callbacks=[checkpoint],verbose=1)
with open(model_name+'_model_trained/training_log.txt','a') as f:
    f.write(str(0)+'   '+str(0.0010)+'   :'+str(history.history['loss'])+'   '+str(history.history['accuracy'])+'\n')


