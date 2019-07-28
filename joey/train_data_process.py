import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    #Point out your directory here
    train_dir = '../train_data'
    '''
    full_file, full_file_label = get_full_file_list(train_dir)
    print(len(full_file))
    read_data(full_file[0][1])
    '''
    Datasets = load_data.get_datasets(train_dir)
    le = load_data.label_encoder(train_dir)
    train_data = []
    label_list = []
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
    
    x_train, x_test, y_train, y_test = train_test_split(train_data, label_list, test_size=0.1, random_state=44)
    '''
    print("xtrain ::",x_train[0])
    print("ytrain ::",y_train[0])
    print("xtest ::",x_test[0])
    print("ytest ::",y_test[0])
    '''
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    
    print("Accuracy ::", accuracy)
    
    
if __name__ == '__main__':
    main()
