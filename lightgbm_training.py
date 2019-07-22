import numpy as np
import load_data
import preprocess as pre
import features

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb


def get_features(dataset_dir):
    # Load data
    print(">Loading data")
    dataset, le = load_data.get_datasets_with_encoder(dataset_dir)

    # Data preprocessing
    print(">Preprocessing data")
    fixed_d = pre.non_zero_fix(dataset)
    fixed_d = pre.interpolation_fix(fixed_d)
    fixed_d = pre.huge_change_fix(fixed_d)

    # Extract features
    print(">Extracting features")
    train_data = features.starting_value_mean(datasets=fixed_d, length=200)
    #train_data = features.top_temper(datasets=fixed_d, num_col=8)
    #train_data = features.top_col_mean_temper(datasets=fixed_d, length=150)
    #train_data = features.top_col_stack_temper(datasets=fixed_d, length=150)
    label_list = le.transform([d.label for d in dataset])
    print('Feature shape= ')
    print(train_data.shape)
    return train_data, label_list

def trainer_lightgbm(train_data, label_list, max_depth, num_leaves, seed=42, debug=True):
    x_train, x_test, y_train, y_test = train_test_split(train_data, label_list, test_size=0.3, random_state=seed)

    ''' 
    clf = lgb.LGBMClassifier(silent= not debug
            , max_depth=5
            , num_leaves=15)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    '''
    clf = lgb.LGBMClassifier(silent= not debug
            , max_depth=max_depth
            , num_leaves=num_leaves)
    bag = BaggingClassifier(base_estimator=clf, max_samples=0.5, max_features=0.5)
    bag.fit(x_train, y_train)
    y_predict = bag.predict(x_test)
    
    #print(y_predict)
    #print(y_test)
    accuracy = accuracy_score(y_predict, y_test)

    print("Accuracy ::", accuracy)
    return accuracy

'''
Plot a 2D bar, count the appearing frequency
Param:
    arr: 1D list,
    slices: slice to n level
'''
def plot_counting(arr, slices=10):
    arr = sorted(arr)
    scope = np.linspace(arr[0], 1, num=slices)
    count = []
    index = 0
    for s in scope:
        sub = 0
        while index < len(arr) and arr[index] <= s:
            sub += 1
            index += 1
        print('below ' + str(s) + ' :' + str(sub) )
        count.append(sub)

    plt.figure()
    plt.bar(scope, count, width=(arr[-1]-arr[0])/slices)
    #plt.plot(scope, count)
    plt.show()

'''
Plot a 3D bar
Param:
    arr: 2d list, 
    pa1: boarders of first axis,
    pa2: boarders of second axis,
'''
def plot_counting_3d(arr, pa1, pa2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for acc, y in zip(arr, pa1):
        ax.bar(pa2, acc, zs=y, zdir='y', alpha=0.8)
    mini = np.min(np.array(arr))
    ax.set_zlim(mini, 1)
    ax.set_xlabel('leaves')
    ax.set_ylabel('depth')
    ax.set_zlabel('accuracy')
    plt.show()

def main():
    #print('Available scorer: ')
    #print(sorted(sklearn.metrics.SCORERS.keys()))

    dataset_dir='bigdata_datasets'
    train_data, label_list = get_features(dataset_dir)
    print(">Training")
    depths = [5, 10, 15, 20, 25, 30]
    leaves = [5, 10, 15, 20, 25, 30, 35, 45]
    accuracy = []
    for d in depths:
        leav = []
        for l in leaves:
            
            acc = trainer_lightgbm(train_data, 
                    label_list, 
                    max_depth=d, 
                    num_leaves=l, 
                    seed=1, 
                    debug=False)
            
            #acc = random.random()
            leav.append(acc)
        accuracy.append(leav)
    #plot_counting(accuracy)
    plot_counting_3d(accuracy, depths, leaves)
    
    '''
    x_train, x_test, y_train, y_test = train_test_split(train_data, label_list, test_size=0.3, random_state=42)
    clf = lgb.LGBMClassifier(silent=True
            , max_depth=10
            , num_leaves=40)
    bag = BaggingClassifier(base_estimator=clf, max_samples=0.5, max_features=0.5)
    #bag.fit(x_train, y_train)

    param_grid = {
            'max_depth':[5, 10, 15, 20],
            'num_leaves':[20, 25, 30, 35, 40]
    }
    gbm = GridSearchCV(bag
            , param_grid=param_grid
            , scoring='accuracy'
            , cv=3)
    gbm.fit(x_train, y_train)
    
    print('Best parameters found by grid search are:', gbm.best_params_)
    
    y_predict = gbm.predict(x_test)
    print(y_predict)
    print(y_test)
    accuracy = accuracy_score(y_predict, y_test)
    
    print("Accuracy ::", accuracy)
    '''

if __name__ == '__main__':
    main()
