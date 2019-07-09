import load_data
import matplotlib.pyplot as plt


"""
Plot label with specified index.
This funx will plot each column of different samples in a figure.
Para:
    datasets(array): array element with 'datalist' and 'label'
    index(int): specified index
"""
def plot_spec_label(datasets, index):
    _plot_spec_label(datasets, index)
    plt.show()

def _plot_spec_label(datasets, index):
    PLOT_INDEX = index
    NUM_COL = 8
    PLOT_LABEL = datasets[PLOT_INDEX].label
    plt.figure(PLOT_LABEL, figsize=(20,3))
    for i in range(NUM_COL):
        fig_name = str(PLOT_LABEL)+"col_"+str(i)
        plt.subplot(1, NUM_COL, i+1)
        for d in datasets:
            if d.label == PLOT_LABEL:
                try:
                    plt.plot(range(len(d.datalist[i])), d.datalist[i])
                except IndexError:
                    break;
        plt.title(fig_name)

"""
Plot all label, automatically jump to next label.
"""
def plot_all(datasets):
    pre_label = ""
    for i in range(len(datasets)):
        if datasets[i].label != pre_label:
            print("Ploting index "+str(i))
            _plot_spec_label(datasets, i)
            pre_label = datasets[i].label

    plt.show()

def main():
    DATASET_DIR = './bigdata_datasets'
    datasets = load_data.get_datasets(DATASET_DIR)
    plot_all(datasets)


if __name__ == '__main__':
    main()
