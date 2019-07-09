import numpy as np
import load_data
import plotter

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

def main():
    DATASET_DIR = './bigdata_datasets'
    datasets = load_data.get_datasets(DATASET_DIR)
    fixed_d = non_zero_fix(datasets)
    fixed_d = interpolation_fix(fixed_d)
    plotter.plot_all(fixed_d)

if __name__ == '__main__':
    main()