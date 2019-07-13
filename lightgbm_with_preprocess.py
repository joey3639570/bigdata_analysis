import numpy as np
import load_data
import preprocess as pre

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

def first150_with_preprocess(Datasets, train_dir):
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

def first150_with_preprocess_encoder(Datasets, le):
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


def main():
    dataset_dir='bigdata_datasets'
    #dataset = load_data.get_datasets(dataset_dir)
    dataset, le = load_data.get_datasets_with_encoder(dataset_dir)

    fixed_d = pre.non_zero_fix(dataset)
    fixed_d = pre.interpolation_fix(fixed_d)
    # fixed_d = huge_change_fix(fixed_d)
    train_data ,label_list = first150_with_preprocess_encoder(fixed_d, le)
    
    x_train, x_test, y_train, y_test = train_test_split(train_data, label_list, test_size=0.3, random_state=42)
    
    clf = lgb.LGBMClassifier()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print(y_predict)
    print(y_test)
    accuracy = accuracy_score(y_predict, y_test)
    
    print("Accuracy ::", accuracy)
    
if __name__ == '__main__':
    main()
