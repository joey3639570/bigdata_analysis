import numpy as np

"""
Return the mean of top 'length' temperature each column seperately
Specify 'start' can skip the starting numbers
"""
def top_temper(datasets, start=5, length=30):
    new_d = []
    for d in datasets:
        newDataset = load_data.dataset()
        
        sorted_arr = np.sort(d.datalist)
        arr_len = sorted_arr.shape[1]
        fea = sorted_arr[:,arr_len-length-start:arr_len-start] # Get the maximum
        fea = np.mean(fea, axis=1)
        #print(d.label)
        #print(fea)
        newDataset.datalist = fea
        newDataset.label = d.label
        new_d.append(newDataset)
    return np.array(new_d)

def main():
    DATASET_DIR = './bigdata_datasets'
    datasets = load_data.get_datasets(DATASET_DIR)
    fixed_d = non_zero_fix(datasets)
    fixed_d = interpolation_fix(fixed_d)
    fea = top_temper(fixed_d))

if __name__ == '__main__':
    main()
