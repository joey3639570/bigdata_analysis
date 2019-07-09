import dataset_setup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import lightgbm as lgb

def main():
    train_dir = '../train_data'
    train_data ,label_list = dataset_setup.first150_with_bad_value(train_dir)
    
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