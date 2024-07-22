from dataset_informations import partitioner
import numpy as np
class batcher(partitioner):
    def __init__(self, directories, tree_name,batch_size=64,training_size=0.5,testing_size=0.3,validation_size=0.2,shuffle=False):
        self.dir=directories
        self.tn=tree_name
        
        self.bs=batch_size
        
        self.tr_s=training_size
        self.te_s=testing_size
        self.va_s=validation_size

        self.shuffle=shuffle
        
        super().__init__(self.dir,self.tn,self.tr_s,self.te_s,self.va_s,self.shuffle)
        dataset=super().get_train_test_validation()
        
        self.train_set=dataset[0]
        self.test_set=dataset[1]
        self.validation_set=dataset[2]
        
        self.tot_tr=0
        self.tot_te=0
        self.tot_va=0
        
        for dir in self.set_per_dir:
            self.tot_tr+=dir[0]
            self.tot_te+=dir[1]
            self.tot_va+=dir[2]

        self.batch_per_dir=[]

        for dir in self.set_per_dir:
            tr_batch=int(np.ceil((dir[0]/self.tot_tr)*self.bs))
            te_batch=int(np.ceil((dir[1]/self.tot_te)*self.bs))
            va_batch=int(np.ceil((dir[2]/self.tot_va)*self.bs))
            self.batch_per_dir.append([tr_batch,te_batch,va_batch])


        #definiamo il numero di batch in base alla directory che ne contiene il numero minore 
        #per ogni set

        
        self.num_batches=[-1,-1,-1]
        for set,set_ev in enumerate(self.set_per_dir):
            for set_batch,ev_in_batch in enumerate(self.batch_per_dir[set]):
                if self.num_batches[set_batch]==-1 or self.num_batches[set_batch]>set_ev[set_batch]/ev_in_batch:
                    self.num_batches[set_batch]=int(np.ceil(set_ev[set_batch]/ev_in_batch))

    def train_test_validation_batcher(self):
        self.batches_fl_tr=[]
        self.batches_ev_tr=[]

        self.batches_fl_te=[]
        self.batches_ev_te=[]

        self.batches_fl_va=[]
        self.batches_ev_va=[]

        for i in range(self.num_batches[0]):
            self.batches_fl_tr.append([])
            self.batches_ev_tr.append([])

        for i in range(self.num_batches[1]):
            self.batches_fl_te.append([])
            self.batches_ev_te.append([])

        for i in range(self.num_batches[2]):
            self.batches_fl_va.append([])
            self.batches_ev_va.append([])
    
        for i,dir in enumerate(self.train_set[0]):
            starting_file=0
            starting_ev=self.train_set[1][i][0][0]
            for batch_idx in range(self.num_batches[0]):
                ev_in_batch=0
                batch_complete=False
                for j in range(starting_file,len(dir)):
                    file=dir[j]
                    for ev_idx in range(starting_ev,self.train_set[1][i][j][1]):
                        ev_in_batch+=1
                        if ev_in_batch==self.batch_per_dir[i][0]:
                            batch_complete=True
                            self.batches_fl_tr[batch_idx].append(file)
                            self.batches_ev_tr[batch_idx].append([starting_ev,ev_idx+1])
                            starting_ev=ev_idx+1
                            starting_file=j
                            break
                    if batch_complete:
                        break
                    else:
                        self.batches_fl_tr[batch_idx].append(file)
                        self.batches_ev_tr[batch_idx].append([starting_ev,ev_idx+1])
                        if j<(len(dir)-1):
                            starting_ev=self.train_set[1][i][j+1][0]
                        else:
                            break

        for i,dir in enumerate(self.test_set[0]):
            starting_file=0
            starting_ev=self.test_set[1][i][0][0]
            for batch_idx in range(self.num_batches[1]):
                ev_in_batch=0
                batch_complete=False
                for j in range(starting_file,len(dir)):
                    file=dir[j]
                    for ev_idx in range(starting_ev,self.test_set[1][i][j][1]):
                        ev_in_batch+=1
                        if ev_in_batch==self.batch_per_dir[i][1]:
                            batch_complete=True
                            self.batches_fl_te[batch_idx].append(file)
                            self.batches_ev_te[batch_idx].append([starting_ev,ev_idx+1])
                            starting_ev=ev_idx+1
                            starting_file=j
                            break
                    if batch_complete:
                        break
                    else:
                        self.batches_fl_te[batch_idx].append(file)
                        self.batches_ev_te[batch_idx].append([starting_ev,ev_idx+1])
                        if j<(len(dir)-1):
                            starting_ev=self.test_set[1][i][j+1][0]
                        else:
                            break

        for i,dir in enumerate(self.validation_set[0]):
            starting_file=0
            starting_ev=self.validation_set[1][i][0][0]
            for batch_idx in range(self.num_batches[2]):
                ev_in_batch=0
                batch_complete=False
                for j in range(starting_file,len(dir)):
                    file=dir[j]
                    for ev_idx in range(starting_ev,self.validation_set[1][i][j][1]):
                        ev_in_batch+=1
                        if ev_in_batch==self.batch_per_dir[i][2]:
                            batch_complete=True
                            self.batches_fl_va[batch_idx].append(file)
                            self.batches_ev_va[batch_idx].append([starting_ev,ev_idx+1])
                            starting_ev=ev_idx+1
                            starting_file=j
                            break
                    if batch_complete:
                        break
                    else:
                        self.batches_fl_va[batch_idx].append(file)
                        self.batches_ev_va[batch_idx].append([starting_ev,ev_idx+1])
                        if j<(len(dir)-1):
                            starting_ev=self.validation_set[1][i][j+1][0]
                        else:
                            break

        return [self.batches_fl_tr,self.batches_ev_tr],[self.batches_fl_te,self.batches_ev_te],[self.batches_fl_va,self.batches_ev_va] 