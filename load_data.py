import pandas as pd
import numpy as np
import os

# Put the directory of data in, you can get the full file list of the directory
def get_full_file_list(train_dir):
    full_file_list = []
    group_dir = os.listdir(train_dir)
    for group in group_dir:
        file_list = []
        #print(group)
        path = train_dir + '/' + group
        #print(os.listdir(path))
        file_array = os.listdir(path)
        for file in file_array:
            file_path = path + '/' + file
            file_list.append(file_path)
        full_file_list.append(file_list)
    print("Get the full file list...")
    #print(full_file_list)
    return full_file_list
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
def read_data(full_file_path):
    #Using tab to seperate data
    data = pd.read_csv(full_file_path, sep="\t")
    '''
    print(data)
    print(len(data.columns))
    print(data.columns)
    '''
    nparray_data = []
    print("There are", len(data.columns)-1 , "rows.")
    for i in range(0,len(data.columns)-1):
        array_data = data.iloc[1:,i].to_numpy()
        #Get rid of the space
        array_data = np.array([s.strip() for s in array_data],dtype=np.float32)
        nparray_data.append(array_data)
        #print(array_data)
        #print(len(array_data))
    nparray_data = np.stack(nparray_data,axis=0)
    print(nparray_data)
    
    
        
def main():
    #Point out your directory here
    train_dir = 'train_data'
    full_file = get_full_file_list(train_dir)
    print(full_file[0][1])
    read_data(full_file[0][1])
    
if __name__ == '__main__':
    main()

