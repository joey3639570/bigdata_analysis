import dataset_setup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import lightgbm as lgb

def main():
    train_dir = '../bigdata_analysis/train_data'
    train_data ,label_list = dataset_setup.first150_with_bad_value(train_dir)
    maximum=[]
    print(len(train_data))

    for i in range(0,len(train_data)):
        temp = []
        for j in range(0,4):
            for k in range(j*150,j*150+150):
                if train_data[i][k]>train_data[i][k-1]:
                    temp_max=train_data[i][k]
            for k in range(j*150,j*150+150):
                temp.append(temp_max)
        temp = np.array(temp,dtype=np.float32)
        maximum.append(temp)
    '''
    print("FFFFFFFFF")
    print(train_data[7])
    print("AAAAAAAAA")
    print(maximum)
    '''
    x=np.hstack((train_data, maximum))
    x_train, x_test, y_train, y_test = train_test_split(x, label_list, test_size=0.3, random_state=0)
    
    clf = lgb.LGBMClassifier()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print(y_predict)
    print(y_test)
    accuracy = accuracy_score(y_predict, y_test)
    print("Accuracy ::", accuracy)
    
    x_train, x_test, y_train, y_test = train_test_split(train_data, label_list, test_size=0.3, random_state=7)
    clf = lgb.LGBMClassifier()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print(y_predict)
    print(y_test)
    accuracy = accuracy_score(y_predict, y_test)
    print("Accuracy ::", accuracy)
    
if __name__ == '__main__':
    main()