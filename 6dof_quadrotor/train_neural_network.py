import numpy as np
from neural_network import *
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import pickle
from dataset_handler import load_training_dataset

from parameters.octorotor_parameters import num_rotors


def train_neural_network():
    global trainval_dataset

    # 1. Loading datasets and creating dataloaders (old way) #######################
    # global_dataset_preload = ControlAllocationDataset_Binary(datasets_folder, False, num_outputs)
    # global_dataset = ControlAllocationDataset_Binary_Short(global_dataset_preload.get_dataset(), num_outputs)
    # train_size = int(0.8 * len(global_dataset))
    # val_size = int(0.1 * len(global_dataset))
    # test_size = len(global_dataset) - train_size - val_size

    # train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(global_dataset, [train_size, val_size, test_size])

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True if device == 'cuda' else False)

    # validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True if device == 'cuda' else False)

    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True if device == 'cuda' else False)
    ##################################################################################

    # Split into training and validation sets without copying:
    indexes = np.arange(len(trainval_dataset))
    np.random.shuffle(indexes)
    train_ratio = 1 - 0.11
    split = int(train_ratio * len(indexes))
    train_indices = indexes[:split]
    val_indices = indexes[split:]

    train_dataset = trainval_dataset[train_indices].copy()
    val_dataset = trainval_dataset[val_indices].copy()
    del trainval_dataset
    del train_indices
    del val_indices

    # Use samplers to select subsets
    #train_sampler = SubsetRandomSampler(train_indices)
    #val_sampler = SubsetRandomSampler(val_indices)

    # Create loaders with samplers
    pin_memory = True if device == 'cuda' else False
    num_workers = 1

    train_dataset_class = ControlAllocationDataset_Binary_Short(train_dataset, num_rotors)
    val_dataset_class = ControlAllocationDataset_Binary_Short(val_dataset, num_rotors)

    train_dataloader = DataLoader(train_dataset_class, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    validation_dataloader = DataLoader(val_dataset_class, batch_size=batch_size,shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    earlystopper = EarlyStopper(4, 15, 0.05, datasets_folder)

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for i_batch, batch_sample in enumerate(train_dataloader):
            input = batch_sample['input'].to(device, non_blocking = True)
            output = batch_sample['output'].to(device, non_blocking = True)
            #print("input device:", input.device)
            #print("model param device:", next(model.parameters()).device)
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input.size(0)
        train_loss = running_loss / len(train_dataloader.dataset)

        # Validation loop
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i_batch, batch_sample in enumerate(validation_dataloader):
                input = batch_sample['input'].to(device, non_blocking = True)
                output = batch_sample['output'].to(device, non_blocking = True)
                outputs = model(input)
                loss = criterion(outputs, output)
                running_loss += loss.item() * input.size(0)
        val_loss = running_loss / len(validation_dataloader.dataset)

        end_time = time.time()

        training_metadata['execution_time'].append(end_time - start_time)
        training_metadata['epoch'].append(epoch + 1)
        training_metadata['train_loss'].append(train_loss)
        training_metadata['validation_loss'].append(val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}')

        if earlystopper.stop_early(val_loss, model):
            print('Stopped early to avoid overfitting')
            break

    # Test loss (deprecated) #####################################################
    # if optuna_version == 'v2':
    #     model = NeuralNetwork_optuna2(num_inputs, num_rotors).to(device)
    # if optuna_version == 'v3':
    #     model = NeuralNetwork_optuna3(num_inputs, num_rotors).to(device)
    # if optuna_version == 'v4':
    #     model = NeuralNetwork_optuna4(num_inputs, num_rotors).to(device)
    #     model.load_state_dict(torch.load(datasets_folder + 'model_weights.pth', weights_only=True))
    #     model.eval()
    #     test_loss = 0.0
    #     with torch.no_grad():
    #         for i_batch, batch_sample in enumerate(test_dataloader):
    #             input = batch_sample['input'].to(device, non_blocking = True)
    #             output = batch_sample['output'].to(device, non_blocking = True)
    #             outputs = model(input)
    #             loss = criterion(outputs, output)
    #             test_loss += loss.item() * batch_sample['input'].size(0)            
    #     test_loss = test_loss / len(test_dataloader.dataset)
    #     with open(datasets_folder + "test_losses.txt", "w") as f:
    #         f.write(str(test_loss))
    # else: print('Test loss not saved. mismatch of optuna versions!!!')
    ############################################################################

    trim_idx = np.min([len(training_metadata['epoch']), len(training_metadata['train_loss']), len(training_metadata['validation_loss'])])
    training_dataframe = pd.DataFrame(training_metadata)
    training_dataframe.to_csv(datasets_folder + 'training_metadata.csv', sep = ',', index=False)

    fig = plt.figure()
    x = training_metadata['epoch']
    plt.plot(x, training_metadata['train_loss'][:trim_idx])
    plt.plot(x, training_metadata['validation_loss'][:trim_idx])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Train and Test Losses')
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.savefig(datasets_folder + 'losses.png')
    plt.savefig(datasets_folder + 'losses.pdf')
    plt.show()

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 1000
    batch_size = 128
    #learning_rate = 0
    num_outputs = num_rotors
    optuna_version = 'v6_database_v5_smooth_kfold'
    # Dataset path
    datasets_folder = '../Datasets/Training datasets - v5/'

    print('CUDA:', torch.cuda.is_available(), torch.accelerator.is_available())
    print(torch.cuda.is_available())  # Should print True
    print(torch.cuda.device_count())  # Should be > 0
    print(torch.cuda.get_device_name(0)) 


    load_previous_model = False
    previous_model_path = ''

    training_metadata = {
        'epoch': [],
        'train_loss': [],
        'validation_loss': [],
        'execution_time': []
    }    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    
    # 1. Loading datasets and creating dataloaders
    trainval_dataset, num_inputs = load_training_dataset(datasets_folder, num_rotors)

    ### 2. Building the Neural Network ###
    model = NeuralNetwork(optuna_version, num_inputs, num_outputs).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = getattr(torch.optim, model.optimizer)(model.parameters(), lr = model.opt_leaning_rate, weight_decay=model.l2_lambda)

    train_neural_network()

    # 
    # 
    #  outputs = model(x.unsqueeze(1))
    #     loss = criterion(outputs.squeeze(), f(x))
    #     loss.backward()
    #     optimizer.step()
    #     running_loss += loss.item()

    #     if epoch % 100 == 0:
    #     print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

