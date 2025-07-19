import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'Group individual datasets, split dataset into training and test'

def group_and_split_datasets_npy(folder_path):
    '''
    Groups all the npy dataset files inside folder_path and splits it into training and test datasets. The training set is normalized and the normalization data is saved in a csv
    '''
    #https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/

    global_dataset = []

    print('Loading dataset indexes...')
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file == 'dataset.npy':
                #self.npy_files.append(os.path.join(subdir, file))
                global_dataset.append(np.load(os.path.join(subdir, file)))
    global_dataset = np.concatenate(global_dataset, axis = 0)
    #num_inputs = len(global_dataset[0]) - num_outputs
    
    print(f'\tLoaded {len(global_dataset)} samples')
    print(f'\tSample length: {len(global_dataset[0])}')
    print(f'\tDataset size: {global_dataset.nbytes / 1024**3} MB')
    print('Interity check:')
    for i in range(len(global_dataset)):
        if len(global_dataset[i]) != len(global_dataset[0]):
            print(f'corrupted row: i={i}, {len(global_dataset[i])} samples')
    print('\t There are no corrupted data')
    raise Exception('Protection')
    # Spliting into train and test (obs: validation happens inside the training dataset)
    training_dataset, test_dataset = train_test_split(global_dataset, test_size=0.1, random_state=50, shuffle=True)

    # Normalization of the training dataset ONLY
    mean = np.mean(training_dataset, axis = 0)
    std = np.std(training_dataset, axis = 0)

    training_dataset = (training_dataset - mean) / std

    np.save(folder_path + 'training_split_normalized.npy', training_dataset)
    np.save(folder_path + 'test_split_not_normalized.npy', test_dataset)

    data = np.concatenate(([mean], [std]), axis = 0)
    np.savetxt(folder_path + 'normalization_data.csv', data, delimiter=",")

def group_datasets_csv(folder_path):
    '''
    Groups several csv datasets contained in a folder path into a single file.\n
    folder_path: folder path where all the csv's to be grouped are located
    '''
    #https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/

    global_dataset = []

    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if 'dataset.csv' in file and 'global' not in subdir and 'normalization' not in file:
                dataframe_dir = os.path.join(subdir, file)
                dataframe = pd.read_csv(dataframe_dir, header = None)
                #global_dataset = dataframe if len(dataframe) == 0 else pd.concat((global_dataset, dataframe), axis = 0, ignore_index = True)
                if len(global_dataset) == 0:
                    global_dataset = dataframe
                else:
                    global_dataset = pd.concat((global_dataset, dataframe), axis = 0, ignore_index = True)

    #remover
    # double = global_dataset.astype('float32')
    # print('max', (double - global_dataset).to_numpy().max())
    # print('dffff\n',(global_dataset[179] - double[179]).idxmax())
    # print(global_dataset.iloc[10075,179], double.iloc[10075,179])

    print('Global dataset shape:', np.shape(global_dataset))
    #print('column 50 min max mean', np.min(global_dataset[100]), np.max(global_dataset[100]), np.mean(global_dataset[100]))
    #print('global dataset[98].mean',global_dataset[98].mean())
    #print('global dataset[98].std',global_dataset[98].std())
    
    save_path = folder_path + 'global_dataset/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #global_dataset.to_csv(save_path + 'global_dataset.csv', header = False, index = False)

    # Normalization
    mean_row = []
    std_row = []
    for column in global_dataset.columns:
        mean = global_dataset[column].mean()
        std = global_dataset[column].std()
        global_dataset[column] = (global_dataset[column] - mean)/std
        if global_dataset[column].isnull().values.any():
            print('mean', mean)
            print('std', std)
            print('NaN column', column)

        mean_row.append(mean)
        std_row.append(std)

        np.savetxt(save_path + 'normalization_data.csv', np.array([mean_row, std_row]), delimiter=",")

    #global_dataset.to_csv(save_path + 'global_dataset.csv', header = False, index = False)

def split_dataset(save_folder, dataset_name, train_percentage):
    '''Splits the global dataset into training and test datasets'''
    
    # Read dataframe
    dataframe = pd.read_csv(save_folder + dataset_name, header = None)

    # Shuffling the dataset
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    # Partitioning into training and test datasets
    chop_index = int(np.rint(train_percentage*len(dataframe)))
    training_dataset = dataframe[0:chop_index]
    test_dataset = dataframe[chop_index:]
    print('percentage training dataset =',len(training_dataset)/len(dataframe))
    print('percentage test dataset =',len(test_dataset)/len(dataframe))

    # Saving
    training_dataset.to_csv(save_folder + 'train_dataset.csv', sep=',', header = False, index = False)
    test_dataset.to_csv(save_folder + 'test_dataset.csv', sep=',', header = False, index = False)

def csv2npy(mother_folder_path):
    for subdir, _, files in os.walk(mother_folder_path):
        for file in files:
            if file == 'dataset.csv':
                dataframe_dir = os.path.join(subdir, file)
                dataframe = pd.read_csv(dataframe_dir, header = None)
                data_bin = dataframe.to_numpy(dtype=np.float32)   # Set dtype when converting to NumPy
                np.save(os.path.join(subdir, 'dataset.npy'), data_bin)

def load_dataset(datasets_folder, dataset_name, num_outputs):
    #global_dataset = ControlAllocationDataset_Binary(datasets_folder, False, num_outputs)
    #train_size = int(0.9 * len(global_dataset))
    #test_size = len(global_dataset) - train_size
    #val_size = int(0.2 * len(global_dataset))
    
    #train_dataset, test_dataset = torch.utils.data.random_split(global_dataset, [train_size, test_size])
    #num_inputs = global_dataset.num_inputs

    dataset = np.load(datasets_folder + dataset_name)
    num_inputs = len(dataset[0]) - num_outputs
    print('shape dataset',np.shape(dataset))
    print('number of inputs',num_inputs)
    return dataset, num_inputs

def normalize_test_dataset(datasets_folder):
    test_dataset = np.load(datasets_folder + 'test_split_not_normalized.npy')
    normalized_data = pd.read_csv(datasets_folder + 'normalization_data.csv', sep = ',', header = None)
    print('shape normalized dataset:', np.shape(normalized_data))
    mean = normalized_data.iloc[0, :].values.squeeze()
    std = normalized_data.iloc[1, :].values.squeeze()
    test_dataset = (test_dataset - mean) / std
    np.save(datasets_folder + 'test_split_normalized.npy', test_dataset)


if __name__ == '__main__':
    group_and_split_datasets_npy('../Datasets/Training datasets - v5/')
    #load_test_dataset('../Datasets/Training datasets - v5/')

    # group_datasets('teste/')
    # csv2npy('../Datasets/')
    # split_dataset('dataset_canon/canon_N_90_M_10_hover_only/global_dataset/', 'global_dataset.csv', 0.8)

    # Teste de validação
    # check = pd.read_csv('dataset_canon/canon_N_50_M_20/global_dataset.csv', header = None)
    # print('check global dataset shape',np.shape(check))
    # print('column 50 min max mean', np.min(check[100]), np.max(check[100]), np.mean(check[100]))

    # check_normalization = pd.read_csv('dataset_canon/canon_N_50_M_20/global/normalization_data.csv', header = None)
    # print('check_normalization shape', np.shape(check_normalization))
    # print('check_normalization[98].std\n', check_normalization[98][1])
