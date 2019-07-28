import dataset_setup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
import numpy as np
import lightgbm as lgb
import pandas as pd

def first150_with_preprocessing_average_training(seed,train,label):
    x_train, x_test, y_train, y_test = train_test_split(train, 
                                                        label, 
                                                        test_size=0.3, 
                                                        random_state=seed)
    
    clf = lgb.LGBMClassifier(silent=False)
    bag = BaggingClassifier(base_estimator=clf, max_samples=0.5, max_features=0.5)
    bag.fit(x_train, y_train)
    y_predict = bag.predict(x_test)
    print(y_predict)
    print(y_test)
    accuracy = accuracy_score(y_predict, y_test)
    
    print("Accuracy ::", accuracy)
    return accuracy
    
def main():
    train_dir = '../train_data'
    train_data ,label_list = dataset_setup.first150_with_preprocessing_average(train_dir)
    
    '''
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print(y_predict)
    print(y_test)
    accuracy = accuracy_score(y_predict, y_test)
    
    print("Accuracy ::", accuracy)
    '''
    '''
    param_grid = {
    'boosting_type': ['gbdt','dart','goss'],
    'n_estimators': range(20,100)
    }
    
    gbm = GridSearchCV(clf, param_grid, cv=3)
    gbm.fit(x_train, y_train)
    
    print('Best parameters found by grid search are:', gbm.best_params_)
    
    y_predict = gbm.predict(x_test)
    print(y_predict)
    print(y_test)
    accuracy = accuracy_score(y_predict, y_test)
    
    print("Accuracy ::", accuracy)
    '''
    accuracy = []
    for i in range(1,301):
        print("Random state ::",i)
        acc = first150_with_preprocessing_average_training(seed=i,
                                                     train=train_data,
                                                     label=label_list)
        accuracy.append(acc)
    print(accuracy)
    pd.DataFrame(accuracy).to_excel('output.xlsx', header=False, index=False)
    
if __name__ == '__main__':
    main()