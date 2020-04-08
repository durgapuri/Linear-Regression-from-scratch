#!/usr/bin/env python
# coding: utf-8

# In[164]:


import numpy as np
import pandas as pd
import statistics
import random
from sklearn.metrics import r2_score

class Weather:
    mean_std = list()
    theta_val = None
    alpha = 0.005
    iterations = 1000
    
    def clean_and_prepare_data(self,train_data_frm):
        drop_indices = ['Formatted Date', 'Summary', 'Precip Type', 'Daily Summary']
        train_data_frm = train_data_frm.drop(drop_indices,axis=1)
        predict_col = train_data_frm.iloc[:,1]
        train_data_frm = train_data_frm.drop(["Apparent Temperature (C)"],axis=1)
        train_data_frm = pd.concat([train_data_frm,predict_col],axis=1)
#         print("clean_and_prepare_data" , type(train_data_frm))
#         print(train_data_frm.columns)
        return train_data_frm
    
    def find_mean_std(self,train_data_frm):
        for cols in range(len(train_data_frm.columns)-1):
            col = train_data_frm.iloc[:,cols]
            mean_val = statistics.mean(col)
            std_val = statistics.stdev(col)
            self.mean_std.append([mean_val,std_val])
#         print(self.mean_std)


    def preproccess_data(self,train_data_frm):
        col_list = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
       'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']
        j=0
        for i in col_list:
            train_data_frm[i] = (train_data_frm[i]-self.mean_std[j][0])/self.mean_std[j][1]
            j+=1
        return train_data_frm
    
    def run_gradient_descent(self,train_data_frm):
#         print("set_theta_alpha")
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
#         print(self.theta_val)
    
    def train_validation_split(self,data_frm,validation_data_size):
       
        if isinstance(validation_data_size, float):
            validation_data_size=round(validation_data_size * len(data_frm))
        indices=data_frm.index.tolist()
        valid_indices=random.sample(indices, validation_data_size)
        valid_datafrm=data_frm.loc[valid_indices]
        train_datafrm=data_frm.drop(valid_indices)
        return train_datafrm, valid_datafrm
    
    def check_validation_data(self,validation_data_frm):
#         validation_data_frm = self.preproccess_data(validation_data_frm)
        actual_data = validation_data_frm.iloc[:,-1].values
        x_data = validation_data_frm.iloc[:,:-1].values
#         print(x_data)
        x_data = np.c_[np.ones(x_data.shape[0]),x_data]
        predicted_data = x_data.dot(self.theta_val)
#         print(actual_data.shape)
#         print(predicted_data.shape)
        print(r2_score(actual_data, predicted_data))
    
    def train(self,train_path):
        train_data_frm = pd.read_csv(train_path)
        train_data_frm = self.clean_and_prepare_data(train_data_frm)
        self.find_mean_std(train_data_frm)
        train_data_frm = self.preproccess_data(train_data_frm)
        random.seed(0)
        train_data_frm, validation_data_frm = self.train_validation_split(train_data_frm, validation_data_size = 0.4)
        self.run_gradient_descent(train_data_frm)
#         self.check_validation_data(validation_data_frm)
#         print(train_data_frm)

    def predict(self,test_path):
        test_data_frm = pd.read_csv(test_path)
        drop_indices = ['Formatted Date', 'Summary', 'Precip Type', 'Daily Summary']
        test_data_frm = test_data_frm.drop(drop_indices,axis=1)
        test_data = self.preproccess_data(test_data_frm).values
        test_data = np.c_[np.ones(test_data.shape[0]),test_data]
        predict_values = test_data.dot(self.theta_val)
        return predict_values
    
        


# In[163]:


# model4 = Weather()
# model4.train('/home/jyoti/Documents/SMAI/assign2/Q4/Question-4/weather.csv') # Path to the train.csv will be provided
# prediction4 = model4.predict('/home/jyoti/Documents/SMAI/assign2/Q4/Question-4/weather_test.csv') 
# print(prediction4)


# In[ ]:





# In[ ]:




