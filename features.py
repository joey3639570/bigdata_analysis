import numpy as np

def starting_value_mean(datasets, length=150):
    data = []
    for d in datasets:
        fea = d.datalist[:,0:length]
        fea = np.mean(fea, axis=0)
        if len(fea) < length:
            compen = fea[-1]*np.ones(length-len(fea))
            fea = np.append(fea, compen)
        data.append(fea)
    data = np.stack(data)
    return data

def starting_value_stack(datasets, length=150):
    data = []
    for d in datasets:
        fea = d.datalist[:,0:length]
        fea = fea[0:4].reshape(4*length)
        if len(fea) < length:
            compen = fea[-1]*np.ones(length-len(fea))
            fea = np.append(fea, compen)
        data.append(fea)
    data = np.stack(data)
    return data

"""
Return the mean of top 'length' temperature each column seperately.
Specify 'start' can skip the starting numbers.
It also fixed the defective column.
"""
def top_temper(datasets, num_col, start=5, length=30):
    data = []
    for d in datasets:
        sorted_arr = np.sort(d.datalist)
        arr_len = sorted_arr.shape[1]
        fea = sorted_arr[:,arr_len-length-start:arr_len-start]
        fea = np.mean(fea, axis=1)
        # Compensate the defective array with the average value
        while len(fea) < num_col:
            fea = np.append(fea, np.mean(fea))
        data.append(fea)
    data = np.stack(data)
    return data

def top_temper(datasets, num_col, start=5, length=30):
    data = []
    for d in datasets:
        sorted_arr = np.sort(d.datalist)
        arr_len = sorted_arr.shape[1]
        fea = sorted_arr[:,arr_len-length-start:arr_len-start]
        fea = np.mean(fea, axis=1)
        # Compensate the defective array with the average value
        while len(fea) < num_col:
            fea = np.append(fea, np.mean(fea))
        data.append(fea)
    data = np.stack(data)
    return data

def top_col_mean_temper(datasets, start=5, length=150):
    data = []
    for d in datasets:
        sorted_arr = np.sort(d.datalist)
        arr_len = sorted_arr.shape[1]
        fea = sorted_arr[:,arr_len-length-start:arr_len-start]
        fea = np.mean(fea, axis=0)
        data.append(fea)
    data = np.stack(data)
    return data

def top_col_stack_temper(datasets, start=5, length=150):
    data = []
    for d in datasets:
        sorted_arr = np.sort(d.datalist)
        arr_len = sorted_arr.shape[1]
        fea = sorted_arr[:,arr_len-length-start:arr_len-start]
        fea = fea[0:4].reshape(4*length)
        data.append(fea)
    data = np.stack(data)
    return data

def main():
    DATASET_DIR = './bigdata_datasets'
    datasets = load_data.get_datasets(DATASET_DIR)
    fixed_d = non_zero_fix(datasets)
    fixed_d = interpolation_fix(fixed_d)
    fea = top_temper(fixed_d)

if __name__ == '__main__':
    main()
