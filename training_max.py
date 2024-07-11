import numpy as np
import pickle
import random
import pandas as pd
import uproot as up
import gc

import warnings
warnings.filterwarnings('ignore')


with open('Dataset_information/electrons_100GeV_1TeV_path_size','rb') as f:
    electrons_100GeV_1TeV_path_size=pickle.load(f)
with open('Dataset_information/electrons_1TeV_20TeV_path_size','rb') as f:
    electrons_1TeV_20TeV_path_size=pickle.load(f)
with open('Dataset_information/protons_100GeV_1TeV_path_size','rb') as f:
    protons_100GeV_1TeV_path_size=pickle.load(f)
with open('Dataset_information/protons_1TeV_10TeV_path_size','rb') as f:
    protons_1TeV_10TeV_path_size=pickle.load(f)

print(electrons_100GeV_1TeV_path_size)


electrons_100GeV_1TeV_path=[row[0] for row in electrons_100GeV_1TeV_path_size]
electrons_1TeV_20TeV_path=[row[0] for row in electrons_1TeV_20TeV_path_size]
protons_100GeV_1TeV_path=[row[0] for row in protons_100GeV_1TeV_path_size]
protons_1TeV_10TeV_path=[row[0] for row in protons_1TeV_10TeV_path_size]

electrons_100GeV_1TeV_size=[row[1] for row in electrons_100GeV_1TeV_path_size]
electrons_1TeV_20TeV_size=[row[1] for row in electrons_1TeV_20TeV_path_size]
protons_100GeV_1TeV_size=[row[1] for row in protons_100GeV_1TeV_path_size]
protons_1TeV_10TeV_size=[row[1] for row in protons_1TeV_10TeV_path_size]

def training_test_split(size,train_test_ratio):
    somma=0
    train_size=sum(size)*train_test_ratio
    for i in range(len(size)):
        somma+=size[i]
        if somma >= train_size:
            return i

final_electrons_100GeV_1TeV_training_file=training_test_split(electrons_100GeV_1TeV_size,0.5)
final_electrons_1TeV_20TeV_training_file=training_test_split(electrons_1TeV_20TeV_size,0.5)
final_protons_100GeV_1TeV_training_file=training_test_split(protons_100GeV_1TeV_size,0.5)
final_protons_1TeV_10TeV_training_file=training_test_split(protons_1TeV_10TeV_size,0.5)


electrons_100GeV_1TeV_path_tree=[name+':showersTree;1' for name in electrons_100GeV_1TeV_path]
electrons_1TeV_20TeV_path_tree=[name+':showersTree;1' for name in electrons_1TeV_20TeV_path]

protons_100GeV_1TeV_path_tree=[name+':showersTree;1' for name in protons_100GeV_1TeV_path]
protons_1TeV_10TeV_path_tree=[name+':showersTree;1' for name in protons_1TeV_10TeV_path]

max_values=[]
min_values=[]

def min_max(path_tree,final):
    for file_name in path_tree[:final+1]:
        for df in up.iterate(file_name,'deps2D',library="pd", step_size='500MB'):
            max_values.append(df.max())
            min_values.append(df.min())
            del df
            gc.collect()  

min_max(electrons_100GeV_1TeV_path_tree,final_electrons_100GeV_1TeV_training_file)
min_max(electrons_1TeV_20TeV_path_tree,final_electrons_1TeV_20TeV_training_file)
min_max(protons_100GeV_1TeV_path_tree,final_protons_100GeV_1TeV_training_file)
min_max(protons_1TeV_10TeV_path_tree,final_protons_1TeV_10TeV_training_file)


max_list = [series.values[0] for series in max_values]
min_list = [series.values[0] for series in min_values]

with open('Dataset_information/max_training_values','wb') as f:
    pickle.dump(max(max_list),f)
with open('Dataset_information/min_training_values','wb') as f:
    pickle.dump(min(min_list),f)
