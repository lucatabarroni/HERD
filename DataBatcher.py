from keras.utils import Sequence
import uproot as up
import numpy as np
import pandas as pd

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


class BatcherCreation(Batcher,Sequence):
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
        
        self.dataset=pd.DataFrame()
        self.labels_per_batch=np.full(self.batch_size,2)
        self.labels=np.full((self.min_len,self.batch_size),self.labels_per_batch)

        for i in range(self.min_len):
            self.labels[i]=np.concatenate((np.ones(self.low_electrons_per_batch),np.ones(self.high_electrons_per_batch),np.zeros(self.low_protons_per_batch),np.zeros(self.high_protons_per_batch)))
        
    def __len__(self):
        return self.min_len
    
    def creation_dataset(self,idx):
        matr_batch=np.full((self.batch_size,20,20),2,dtype=float)
        for i in range(self.batch_size):
            matr_batch[i] = np.array(self.dataset.iloc[i*400:(i+1)*400]).reshape(20, 20)
            matr_batch[i]=matr_batch[i]/self.massimo

        self.dataset=pd.DataFrame()

        indices=np.arange(self.batch_size)
        np.random.shuffle(indices)
        
        matr_batch=matr_batch[indices]
        self.labels[idx]=self.labels[idx][indices]

        return matr_batch.reshape(self.batch_size,20,20),self.labels[idx].reshape(self.batch_size)
    
    def __getitem__(self,idx):
        for low_ele_events in self.low_electrons_batch_per_file[idx]:
            with up.open(low_ele_events[0][:-14]) as file:
                low_ele_df = file['showersTree'].arrays('deps2D', library='pd')[low_ele_events[1]*400:low_ele_events[2]*400]
                self.dataset=pd.concat([self.dataset,low_ele_df])
                del low_ele_df
        
        for high_ele_events in self.high_electrons_batch_per_file[idx]:
            with up.open(high_ele_events[0][:-14]) as file:
                high_ele_df = file['showersTree'].arrays('deps2D', library='pd')[high_ele_events[1]*400:high_ele_events[2]*400]
                self.dataset=pd.concat([self.dataset,high_ele_df])
                del high_ele_df
        
        for low_pro_events in self.low_protons_batch_per_file[idx]:
            with up.open(low_pro_events[0][:-14]) as file:
                low_pro_df = file['showersTree'].arrays('deps2D', library='pd')[low_pro_events[1]*400:low_pro_events[2]*400]
                self.dataset=pd.concat([self.dataset,low_pro_df])
                del low_pro_df
            
        for high_pro_events in self.high_protons_batch_per_file[idx]:
            with up.open(high_pro_events[0][:-14]) as file:
                high_pro_df = file['showersTree'].arrays('deps2D', library='pd')[high_pro_events[1]*400:high_pro_events[2]*400]
                self.dataset=pd.concat([self.dataset,high_pro_df])
                del high_pro_df
        self.dataset=self.dataset.reset_index(drop=True)
        return self.creation_dataset(idx)