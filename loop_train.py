import one_epoch
import numpy as np

learning_rate=0.001
learning_rates=[]
for i in range(50):
    if i%2==0 and i!=6:
      learning_rate=learning_rate*0.95
    learning_rates.append(learning_rate)
starting_epoch=input("Inserire l'epoca da cui partire :" )
model=input("Inserire il modello da trainare (CNN/ResNet_CNN) :")
batch_size=int(input('Scegli un batch size: '))

for i in range(int(starting_epoch),50):
    one_train=one_epoch.One_epoch(batch_size,model)
    one_train.train_one_epoch(i,learning_rates[i])