import pandas as pd
import numpy as np
import os
from sklearn import preprocessing

# Label stands for the group that the data belongs.
class dataset():
    def __init__(self):
        self.datalist = []
        self.label = ''
    
    def read_data(self, full_file_path):
        data = pd.read_csv(full_file_path, sep="\t")
        nparray_data = []
        #print("There are", len(data.columns)-1 , "rows.")
        for i in range(0,len(data.columns)-1):
            array_data = data.iloc[1:,i].to_numpy()
            #Get rid of the space
            array_data = np.array([s.strip() for s in array_data],dtype=np.float32)
            nparray_data.append(array_data)
            #print(array_data)
            #print(len(array_data))
        self.datalist = np.stack(nparray_data,axis=0)
        
    def add_label(self, label):
        self.label = label
    
# Put the directory of data in, you can get the full file list of the directory
def get_full_file_list(train_dir):
    full_file_list = []
    full_file_label_list = []
    group_dir = os.listdir(train_dir)
    for group in group_dir:
        #print(group)
        path = train_dir + '/' + group
        #print(os.listdir(path))
        file_array = os.listdir(path)
        for file in file_array:
            file_path = path + '/' + file
            full_file_list.append(file_path)
            full_file_label_list.append(group)
    print("Get the full file list...")
    print("Get the full file label list...")
    #print(full_file_list)
    #print(full_file_label_list)
    return full_file_list, full_file_label_list

'''
Give a full path of a txt data, 
print out how many rows there are,
return a numpy array of data.

example:
    read_data(train_data/G11/G11-1-AC(7X15)20160311-003_Export.txt)
    
There are 6 rows.
[[ 64.9  64.9  64.9 ... 120.3 118.6 118.5]
 [ 65.2  65.2  65.2 ... 139.3 137.9 137.6]
 [ 65.3  65.3  65.3 ... 129.7 128.9 128.9]
 [ 65.2  65.2  65.2 ... 123.1 121.9 121.8]
 [ 65.   65.   65.  ... 119.6 118.3 118.1]
 [ 64.3  64.3  64.3 ... 118.3 117.6 117.4]]
    
'''
#This function support you for reading data.
def read_data(full_file_path):
    #Using tab to seperate data
    data = pd.read_csv(full_file_path, sep="\t")
    '''
    print(data)
    print(len(data.columns))
    print(data.columns)
    '''
    nparray_data = []
    #print("There are", len(data.columns)-1 , "rows.")
    for i in range(0,len(data.columns)-1):
        array_data = data.iloc[1:,i].to_numpy()
        #Get rid of the space
        array_data = np.array([s.strip() for s in array_data],dtype=np.float32)
        nparray_data.append(array_data)
        #print(array_data)
        #print(len(array_data))
    nparray_data = np.stack(nparray_data,axis=0)
    print(nparray_data)

'''
Give a full path of the train data, 
return a list of dataset objects.

example:
    train_dir = 'train_data'
    Datasets = get_datasets(train_dir)
    print(Datasets[0].datalist)
    print(Datasets[0].label)
    
Get the full file list...
Get the full file label list...
[[ 65.9  65.9  65.9 ... 127.2 125.7 125.3]
 [ 65.8  65.8  65.8 ... 112.9 111.3 110.7]
 [ 64.2  64.2  64.2 ... 111.1 109.6 109.3]
 [ 64.9  64.9  64.9 ... 139.6 138.1 137.5]
 [ 66.   66.   66.  ... 123.  122.  121.8]]
G11

'''
def get_datasets(train_dir):
    """Return a list datasets"""
    file_list, label_list = get_full_file_list(train_dir)
    datasets = []
    for i in range(0,len(file_list)):
        newDataset = dataset()
        newDataset.read_data(file_list[i])
        newDataset.add_label(label_list[i])
        datasets.append(newDataset)
    return datasets

def get_datasets_with_encoder(train_dir):
    datasets = get_datasets(train_dir)
    le = get_label_encoder(datasets)
    return datasets, le
'''
A function for you to make labelencoder, return a labelencoder object
example::
    le = label_encoder(train_dir)
-> Set up labelencoder.
    
    list(le.classes_)
Classes ::
['G11', 'G15', 'G17', 'G19', 'G32', 'G34', 'G48', 'G49']
-> Show you the class of this label encoder

Other example::
    le.transform(["tokyo", "tokyo", "paris"]) 
array([2, 2, 1]...)
-> Allow you to transform new label list

    list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
-> Allow you to transform your result
'''

def label_encoder(train_dir):
    file_list, label_list = get_full_file_list(train_dir)
    le = preprocessing.LabelEncoder()
    le.fit(label_list)
    print("Classes ::")
    print(list(le.classes_))
    return le
        
def get_label_encoder(datasets):
    label_list = [d.label for d in datasets]
    le = preprocessing.LabelEncoder()
    le.fit(label_list)
    return le 

def main():
    #Point out your directory here
    train_dir = 'train_data'
    '''
    full_file, full_file_label = get_full_file_list(train_dir)
    print(len(full_file))
    read_data(full_file[0][1])
    '''
    datasets, le = get_datasets_with_encoder(train_dir)
    result = le.transform(['G11', 'G15', 'G17', 'G19', 'G32', 'G34', 'G48', 'G49'])
    print(result)
    
if __name__ == '__main__':
    main()

