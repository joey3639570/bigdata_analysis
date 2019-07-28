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


def get_features(dataset_dir, mode="train"):
    # Load data
    print(">Loading data")
    dataset, le = load_data.get_datasets_with_encoder(dataset_dir, mode)

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
    if mode=="train":
        return train_data, label_list, le
    else:
        return train_data, label_list

'''
This script will automatically test the model by spliting the train_data and label according to the given test_size.
Param:
    train_data: Include train and test data
    label_list: Include train and test data
    max_depth:  Max depth of the tree in LGBM
    num_leaves: Max number of leaves in LGBM
    seed: The random seed to split test data from train_data
'''
def trainer_lightgbm(train_data, label_list, max_depth, num_leaves, seed=42, debug=True):
    x_train, x_test, y_train, y_test = train_test_split(train_data, label_list, test_size=0.3, random_state=seed)
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

def model_self_test():
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
            
            leav.append(acc)
        accuracy.append(leav)
    #plot_counting(accuracy)
    plot_counting_3d(accuracy, depths, leaves)

'''
Extrade features from train datasets and test datasets.
Each combination of parameters result in a set of answer, and they vote
, to determine the final answer.
The reliability is the ratio of the occurance of the answer.
'''
def main():
    depths = [5, 10, 15, 20, 25, 30]
    leaves = [5, 10, 15, 20, 25, 30, 35, 45]

    print("Getting train data")
    train_data, train_label, le = get_features('bigdata_datasets', mode="train")
    print("Getting test data")
    test_data, test_label = get_features('thubigdata2019exam-722', mode="test")
    
    result = []
    for d in depths:
        for l in leaves:
            clf = lgb.LGBMClassifier(silent= True
                , max_depth=d
                , num_leaves=l)
            bag = BaggingClassifier(base_estimator=clf, max_samples=0.5, max_features=0.5)
            bag.fit(train_data, train_label)
            y_predict = bag.predict(test_data)
            result.append(y_predict)
            print(str(d) + "," + str(l) + " ", end="\t")
            print(y_predict)
            
            #result.append(range(10))

    answers = []
    reliability = []

    result = np.transpose(np.asarray(result))
    sample_length = result.shape[1]
    for ans in result:
        counts = np.bincount(ans)
        answers.append(np.argmax(counts))
        reliability.append(np.amax(counts)/sample_length)
    
    print("Reliability: ")
    print(reliability)
    predict_label = le.inverse_transform(answers)
    print("Answer: ")
    print(predict_label)
    #print(y_test)
    #accuracy = accuracy_score(y_predict, y_test)

if __name__ == '__main__':
    main()
