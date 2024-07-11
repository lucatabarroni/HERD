import numpy as np
import pickle

class LoadData:
    def __init__(self,directory):
        
        self.directory=directory
        
        with open(self.directory,'rb') as f:
            self.path_size=pickle.load(f)
        
        self.path=[row[0] for row in self.path_size]

        self.size=[row[1] for row in self.path_size]

        self.path_tree=[name+':showersTree;1' for name in self.path]

        with open('Dataset_information/max_training_values','rb') as f:
            self.massimo=pickle.load(f)

        with open('Dataset_information/min_training_values','rb') as f:
            self.minimo=pickle.load(f)


    def split_training_test(self,training_test_ratio):
        self.train_size=sum(self.size)*training_test_ratio
        somma=0
        for i in range(len(self.size)):
            somma+=self.size[i]
            if somma >= self.train_size:
                return i

    def training_path_tree(self,training_test_ratio):
        return self.path_tree[0:self.split_training_test(training_test_ratio)+1]

    def training_size(self,training_test_ratio):
        return self.size[0:self.split_training_test(training_test_ratio)+1]

    def testing_path_tree(self,training_test_ratio):
        return self.path_tree[self.split_training_test(training_test_ratio)+1:]

    def testing_size(self,training_test_ratio):
        return self.size[self.split_training_test(training_test_ratio)+1:]
    
        
        
            

