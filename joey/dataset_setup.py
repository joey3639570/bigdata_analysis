import load_data
import numpy as np
import preprocess

'''
This allows you to build a dataset with the first 150 data.
However, this does not kick out the bad value, and it's just for model testing.
Give the directory of your train_dir,
return a 2d numpy array for train data like shown down below,
first data ->[first_colmun second_column third_column fourth_column]
second data ->[first_colmun second_column third_column fourth_column]
a list for label.

example:
    train_dir = '../train_data'
    train,label = first150_with_bad_value(train_dir)
    print(train)
    print(label)
    
Get the full file list...
Get the full file label list...
Get the full file list...
Get the full file label list...
Classes ::
['G11', 'G15', 'G17', 'G19', 'G32', 'G34', 'G48', 'G49']
[[ 65.9  65.9  65.9 ... 354.9 354.8 354.7]
 [ 64.9  64.9  64.9 ... 354.  354.1 354.1]
 [ 75.3  75.3  75.3 ... 354.6 354.6 354.6]
 ...
 [ 74.4  74.4  74.4 ... 349.  349.2 349.4]
 [ 70.9  70.9  70.9 ... 349.8 349.8 350.2]
 [ 76.8  76.8  76.8 ... 339.8 340.2 340.8]]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3
 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
 6 6 6 6 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7
 7 7 7 7 7 7 7 7]
'''

def first150_with_bad_value(train_dir):
    Datasets = load_data.get_datasets(train_dir)
    le = load_data.label_encoder(train_dir)
    train_data = []
    label_list = []
    for dataset in Datasets:
        temp = []
        for i in range(0,4):
            for data in dataset.datalist[i][:150]:
                temp.append(data)
        temp = np.array(temp,dtype=np.float32)
        train_data.append(temp)
        label_list.append(dataset.label)
    
    train_data = np.stack(train_data)
    label_list = le.transform(label_list)
    return train_data, label_list

'''
Give the directory of train data,
return a numpy array train data and a list of label
train data is averaged by combing the temperature column into one.
'''
def first150_with_bad_value_average(train_dir):
    Datasets = load_data.get_datasets(train_dir)
    le = load_data.label_encoder(train_dir)
    train_data = []
    label_list = []
    for dataset in Datasets:
        temp = []
        for data in dataset.datalist:
            temp.append(data[:150])
        temp = np.array(temp,dtype=np.float32)
        temp = np.average(temp, axis=0)
        train_data.append(temp)
        label_list.append(dataset.label)
    
    train_data = np.stack(train_data)
    label_list = le.transform(label_list)
    return train_data, label_list

def first150_with_preprocessing_average(train_dir):
    Datasets = load_data.get_datasets(train_dir)
    fixed_d = preprocess.non_zero_fix(Datasets)
    Datasets = preprocess.interpolation_fix(fixed_d)
    le = load_data.label_encoder(train_dir)
    train_data = []
    label_list = []
    for dataset in Datasets:
        temp = []
        for data in dataset.datalist:
            temp.append(data[:150])
        temp = np.array(temp,dtype=np.float32)
        temp = np.average(temp, axis=0)
        train_data.append(temp)
        label_list.append(dataset.label)
    
    train_data = np.stack(train_data)
    label_list = le.transform(label_list)
    return train_data, label_list

def main():
    train_dir = '../train_data'
    train,label = first150_with_preprocessing_average(train_dir)
    print(train)
    print(label)
    
if __name__ == '__main__':
    main()