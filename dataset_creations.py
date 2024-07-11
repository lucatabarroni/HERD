import numpy as np
import uproot
import os
import pickle

tree_name="showersTree;1"

electrons_100GeV_1TeV_path='/home/rgw/scratch/formato/showerpics/electrons_100GeV_1TeV/'
electrons_100GeV_1TeV_list = [os.path.join(electrons_100GeV_1TeV_path, file) for file in os.listdir(electrons_100GeV_1TeV_path)]
electrons_100GeV_1TeV_list_tree = ["{}:{}".format(file, tree_name) for file in electrons_100GeV_1TeV_list]

electrons_1TeV_20TeV_path='/home/rgw/scratch/formato/showerpics/electrons_1TeV_20TeV/'
electrons_1TeV_20TeV_list = [os.path.join(electrons_1TeV_20TeV_path, file) for file in os.listdir(electrons_1TeV_20TeV_path)]
electrons_1TeV_20TeV_list_tree = ["{}:{}".format(file, tree_name) for file in electrons_1TeV_20TeV_list]

protons_100GeV_1TeV_path='/home/rgw/scratch/formato/showerpics/protons_100GeV_1TeV/'
protons_100GeV_1TeV_list = [os.path.join(protons_100GeV_1TeV_path, file) for file in os.listdir(protons_100GeV_1TeV_path)]
protons_100GeV_1TeV_list_tree = ["{}:{}".format(file, tree_name) for file in protons_100GeV_1TeV_list]

protons_1TeV_10TeV_path='/home/rgw/scratch/formato/showerpics/protons_1TeV_10TeV/'
protons_1TeV_10TeV_list = [os.path.join(protons_1TeV_10TeV_path, file) for file in os.listdir(protons_1TeV_10TeV_path)]
protons_1TeV_10TeV_list_tree = ["{}:{}".format(file, tree_name) for file in protons_1TeV_10TeV_list]


def events_per_file(list,path):
    num_events=[]
    for file in list:
        with uproot.open(os.path.join(path, file)) as f:
            num_events.append((f[tree_name].num_entries))
    return num_events


num_events_electrons_100GeV_1TeV=events_per_file(electrons_100GeV_1TeV_list,electrons_100GeV_1TeV_path)
electrons_100GeV_1TeV_path_size=zip(electrons_100GeV_1TeV_list,num_events_electrons_100GeV_1TeV)
electrons_100GeV_1TeV_path_size=list(electrons_100GeV_1TeV_path_size)
with open('Dataset_information/electrons_100GeV_1TeV_path_size','wb') as file:
    pickle.dump(electrons_100GeV_1TeV_path_size,file)


num_events_electrons_1TeV_20TeV=events_per_file(electrons_1TeV_20TeV_list,electrons_1TeV_20TeV_path)
electrons_1TeV_20TeV_path_size=zip(electrons_1TeV_20TeV_list,num_events_electrons_1TeV_20TeV)
electrons_1TeV_20TeV_path_size=list(electrons_1TeV_20TeV_path_size)
with open('Dataset_information/electrons_1TeV_20TeV_path_size','wb') as file:
    pickle.dump(electrons_1TeV_20TeV_path_size,file)
    

num_events_protons_100GeV_1TeV=events_per_file(protons_100GeV_1TeV_list,protons_100GeV_1TeV_path)
protons_100GeV_1TeV_path_size=zip(protons_100GeV_1TeV_list,num_events_protons_100GeV_1TeV)
protons_100GeV_1TeV_path_size=list(protons_100GeV_1TeV_path_size)
with open('Dataset_information/protons_100GeV_1TeV_path_size','wb') as file:
    pickle.dump(protons_100GeV_1TeV_path_size,file)


num_events_protons_1TeV_10TeV=events_per_file(protons_1TeV_10TeV_list,protons_1TeV_10TeV_path)
protons_1TeV_10TeV_path_size=zip(protons_1TeV_10TeV_list,num_events_protons_1TeV_10TeV)
protons_1TeV_10TeV_path_size=list(protons_1TeV_10TeV_path_size)
with open('Dataset_information/protons_1TeV_10TeV_path_size','wb') as file:
    pickle.dump(protons_1TeV_10TeV_path_size,file)

