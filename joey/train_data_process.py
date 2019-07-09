import load_data
from sklearn import preprocessing


def main():
    #Point out your directory here
    train_dir = '../train_data'
    '''
    full_file, full_file_label = get_full_file_list(train_dir)
    print(len(full_file))
    read_data(full_file[0][1])
    '''
    Datasets = load_data.get_datasets(train_dir)
    '''
    for dataset in Datasets:
        if dataset.label == 'G15':
    '''
    
if __name__ == '__main__':
    main()
