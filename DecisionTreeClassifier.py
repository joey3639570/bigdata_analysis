#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:44:33 2019
Use DecisionTreeClassifier to train in raw-data or preprocess data
@author: linkailun
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import load_data
import preprocess

def main():
    #put datadir here
    train_dir = '../Path_to_Data'
    
    datasets = load_data.get_datasets(train_dir)
    fixed_d = preprocess.non_zero_fix(datasets)
    fixed_d = preprocess.interpolation_fix(fixed_d)
    # IF want preprocess train_data, only comment following two line
    """
    Datasets = fixed_d
    le = load_data.label_encoder(train_dir)
    """
    # IF want use raw data to train, only comment following two line
    """
    Datasets = load_data.get_datasets(train_dir)
    le = load_data.label_encoder(train_dir)
    """
    train_data = []
    label_list = []
    #create label list
    for dataset in Datasets:
        temp = []
        for i in range(0,4):
            for data in dataset.datalist[i][:150]:
                temp.append(data)
            #print(i,len(dataset.datalist[i][:200]))
        temp = np.array(temp,dtype=np.float32)
        #print(temp.shape)
        #print(temp)
        train_data.append(temp)
        label_list.append(dataset.label)
        #print(len(train_data))
        
    train_data = np.stack(train_data)
    #print(train_data.shape)
    #train_data = np.stack(train_data,axis=0)
    label_list = le.transform(label_list)
    # loop for random_state 1 to 100 for test
    for k in range(1,100):
        x_train, x_test, y_train, y_test = train_test_split(train_data, 
                                                            label_list, 
                                                            test_size=0.3, 
                                                            random_state=k)
        
        dt = DecisionTreeClassifier(random_state=0, max_depth=8)
        dt.fit(x_train,y_train)
        
        dt_predict = dt.predict(x_test)
        
        dt_accuracy =accuracy_score(dt_predict, y_test)
        #will print acc every loops
        print(dt_accuracy)

    print("======finish======")
if __name__ == '__main__':
    main()
