# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:54:41 2022

@author: henry
"""

import numpy as np
import scipy.io as sio
import scipy
import torch
import trainer
import ae


for i in range(5):

    mat_contents = sio.loadmat('../sample_data/cardio.mat')
    
    X = mat_contents["X"]
    y = mat_contents["y"]
    
    
    X_normal = X[np.where(y==0)[0]]
    X_abnormal = X[np.where(y==1)[0]]
    
    y_normal = y[np.where(y==0)[0]]
    y_abnormal = y[np.where(y==1)[0]]
    
    print(X_normal.shape)
    print(X_abnormal.shape)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    x_train, x_test, y_train, y_test = train_test_split(X_normal,y_normal,test_size=0.3,random_state=42)
    
    
    x_size = x_train.shape[1]
    mem_dim = 50
    feature_dim = 8
    alpha = 0.001
    epochs = 40000
    
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    model = ae.AE(x_size,mem_dim,feature_dim)
    model_trainer = trainer.Trainer(x_train=scaler.transform(x_train),x_valid=scaler.transform(x_test),model=model,alpha=alpha,batch_size=128,show_progess=1000)
    model_trainer.train(epochs)
    model_trainer.training_loss
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(model_trainer.training_loss[1:],label="normal_train")
    plt.plot(model_trainer.valid_loss[1:],label="normal_valid")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"./experiment/normal_normal/ae_normal_normal_{i+1}_epchs_{epochs}.png")

    with open(f'./experiment/normal_normal/ae_normal_normal_{i+1}_epchs_{epochs}.txt', 'w') as fp:
        for item in zip(model_trainer.training_loss[1:],model_trainer.valid_loss[1:]):
            # write each item on a new line
            fp.write(f"{str(item[0])} {str(item[1])}\n")
        print('Done')