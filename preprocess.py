import numpy as np
import load_data
import plotter
from sklearn.linear_model import LinearRegression

MIN_LIM_TEMP = 20

def differentiate(arr):
    arr_len = len(arr)
    return arr[1:arr_len] - arr[0:arr_len-1]

"""
Replace the value which less than MIN_LIM_TEMP with front nearest value
"""
def non_zero_fix(datasets):
    for d in datasets:
        for col in d.datalist:
            for i in range(len(col)):
                if col[i] < MIN_LIM_TEMP:
                    #Find front nonzero
                    temp_i = i
                    while temp_i > 0 and col[temp_i] < MIN_LIM_TEMP:
                        temp_i -= 1
                    col[i] = col[temp_i]
    return datasets

"""
Find the derivative that have strange overshooting, and replace it with
average value nearby.
"""
def interpolation_fix(datasets):
    for d in datasets:
        for col in d.datalist:
            offset = differentiate(col)
            for i in range(len(offset)):
                if abs(offset[i]) > 10:
                    try:
                        col[i+1] = (col[i] + col[i+2])/2
                    except IndexError:
                        col[i+1] = col[i]
    return datasets

"""
Create noise info
"""
def get_noise(datasets):
    new_d = []
    for d in datasets:
        new_cols = []
        newDataset = load_data.dataset()
        for col in d.datalist:
            offset = differentiate(col)
            new_cols.append(offset)
            #if np.max(abs(offset)) > 20:
            #    print(col)
        newDataset.datalist = np.array(new_cols)
        newDataset.label = d.label
        
        new_d.append(newDataset)    
    return np.array(new_d)
"""
Replace the value which have huge change to last time temperature
This case is over 30 degree, will 
"""
def huge_change_fix(datasets):
    for d in datasets:
        for col in d.datalist:
            for i in range(2,len(col)-1):
                if col[i]-col[i-1] < -30:
                    col[i] = mean_predict(d,col,i)
                elif col[i]-col[i-1] > 30:
                    col[i] = mean_predict(d,col,i)
    return datasets

def huge_change_fix2(datasets):
    count = 0
    b_count = 0
    for d in datasets:
        for col in d.datalist:
            for i in range(6,len(col)-6):
                if col[i]-col[i-1] < -10 and col[i]-col[i-1] > -20:
                    count += 1
                    #print(d.label,"error",col[i])
                    col[i] = linear_predict(col,i)
                elif col[i]-col[i-1] < -20:
                    b_count += 1
                    #print(d.label,"error",col[i])
                    col[i] = mean_predict(d,col,i)
                elif col[i]-col[i-1] > 10 and col[i]-col[i-1] < 20:
                    count += 1
                    #print(d.label,"error",col[i])
                    col[i] = linear_predict(col,i)
                elif col[i]-col[i-1] > 20:
                    b_count += 1
                    #print(d.label,"error",col[i])
                    col[i] = mean_predict(d,col,i)
    print("small",count)
    print("big",b_count)
    return datasets

"""
Linear_regresssion
"""
def linear_predict(col,i):
    X = []
    y = []
    for k in range(i-5,i+5):
        X.append([k])
    #print(X)
    for n in range(i-6,i-1):
        y.append(col[n])
    for n in range(i+1,i+6):
        y.append(col[n])
    #print(y)
    reg = LinearRegression().fit(X, y)
    predict = float(reg.intercept_+reg.coef_*i)
    return predict
"""
Mean predict
Get mean of same time data in that datalist to subtitude the error data
"""

def mean_predict(d,col,i):
    predict = 0
    count=0
    for col in d.datalist:
        predict += col[i]
        count+=1
    count -=1
    predict = predict-col[i]
    predict = predict/count
    return predict

def main():
    DATASET_DIR = './bigdata_datasets'
    datasets = load_data.get_datasets(DATASET_DIR)
    fixed_d = non_zero_fix(datasets)
    fixed_d = interpolation_fix(fixed_d)
    fixed_d = huge_change_fix(fixed_d)
    plotter.plot_all(fixed_d)

if __name__ == '__main__':
    main()
