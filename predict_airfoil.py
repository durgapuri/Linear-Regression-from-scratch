#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import statistics
import random
from sklearn.metrics import r2_score

class Airfoil:
    mean_std = list()
    theta_val = None
    alpha = 0.005
    iterations = 1000
    
    def find_mean_std(self,train_data_frm):
        for cols in range(len(train_data_frm.columns)-1):
            col = train_data_frm.iloc[:,cols]
            mean_val = statistics.mean(col)
            std_val = statistics.stdev(col)
            self.mean_std.append([mean_val,std_val])
        
   
    def preproccess_data(self,train_data_frm):
        for i in range(len(self.mean_std)):
            train_data_frm[i] = (train_data_frm[i]-self.mean_std[i][0])/self.mean_std[i][1]
        return train_data_frm
    
      
    
    def train_validation_split(self,data_frm,validation_data_size):
       
        if isinstance(validation_data_size, float):
            validation_data_size=round(validation_data_size * len(data_frm))
        indices=data_frm.index.tolist()
        valid_indices=random.sample(indices, validation_data_size)
        valid_datafrm=data_frm.loc[valid_indices]
        train_datafrm=data_frm.drop(valid_indices)
        return train_datafrm, valid_datafrm
    
        
    def run_gradient_descent(self,train_data_frm):
        X_data = train_data_frm.iloc[:,:-1].values
        Y_data = train_data_frm.iloc[:,-1].values
        rows, cols = X_data.shape
        X_data = np.c_[np.ones(rows),X_data]
        self.theta_val = np.ones(X_data.shape[1])
        for i in range(self.iterations):
            h = X_data.dot(self.theta_val)
            loss_val = h - Y_data
            gradient = ((X_data.T.dot(loss_val))/X_data.shape[0])
            self.theta_val = self.theta_val - self.alpha * gradient
    
    
    def check_validation_data(self,validation_data_frm):
        actual_data = validation_data_frm.iloc[:,-1].values
        x_data = validation_data_frm.iloc[:,:-1].values
        x_data = np.c_[np.ones(x_data.shape[0]),x_data]
        predicted_data = x_data.dot(self.theta_val)

    
    def train(self,train_path):
        train_data_frm = pd.read_csv(train_path, header=None)
        self.find_mean_std(train_data_frm)
        train_data_frm = self.preproccess_data(train_data_frm)
        random.seed(0)
        train_data_frm, validation_data_frm = self.train_validation_split(train_data_frm, validation_data_size = 0.4)
        self.run_gradient_descent(train_data_frm)

        
    def predict(self,test_path):
        test_data_frm = pd.read_csv(test_path, header=None)
        test_data = self.preproccess_data(test_data_frm).values
        test_data = np.c_[np.ones(test_data.shape[0]),test_data]
        predict_values = test_data.dot(self.theta_val)
        return predict_values
        
        

    


# In[16]:


# model3 = Airfoil()
# model3.train('/home/jyoti/Documents/SMAI/assign2/Q3/airfoil.csv')
# prediction3 = model3.predict('/home/jyoti/Documents/SMAI/assign2/Q3/airfoil_test.csv')
# print(prediction3)


# In[ ]:





# In[ ]:




