import numpy as np
from neural_network import NeuralNetwork, ControlAllocationDataset, ControlAllocationDataset_Binary, EarlyStopper
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import TrialState
from torch import nn
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Hyperparameters
num_epochs = 30
batch_size = 128
#learning_rate = 0.001
num_outputs = 8
num_neurons_hiddenlayers = 128
batches_per_epoch = 250

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#device = torch.device('cpu')
print(f"Using {device} device")

# Dataset path
def load_dataset(datasets_folder):
    global_dataset = ControlAllocationDataset_Binary(datasets_folder, False, num_outputs)
    train_size = int(0.8 * len(global_dataset))
    val_size = len(global_dataset) - train_size
    #val_size = int(0.2 * len(global_dataset))
    
    train_dataset, validation_dataset = torch.utils.data.random_split(global_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True if device == 'cuda' else False)

    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True if device == 'cuda' else False)

    num_inputs = global_dataset.num_inputs

    return train_dataloader, validation_dataloader, num_inputs

def define_model(trial, num_inputs):
    n_layers = trial.suggest_int('n_layers', 1, 5)
    leakyReLU_negative_slope = trial.suggest_float('leakyReLU_negative_slope', 0.0005, 0.1)
    layers = []
    in_features = num_inputs
    for i in range(n_layers):
        out_features = trial.suggest_int('n_units_l{}'.format(i), 64, 2048)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU(negative_slope=leakyReLU_negative_slope))
        #dropout_percentage = trial.suggest_float('dropout_l{}'.format(i), 0.0, 0.5) # Dropout removed, at least temporarily
        #layers.append(nn.Dropout(dropout_percentage))
        in_features = out_features
    
    layers.append(nn.Linear(in_features, num_outputs))

    return nn.Sequential(*layers)


def objective(trial):

    global train_dataloader
    global validation_dataloader
    
    # Generate the model
    model = define_model(trial, num_inputs).to(device)

    # Generate the optimizers
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float('optimizer_learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('l2_lambda', 0, 0.5)
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

    # Loss function
    criterion = torch.nn.MSELoss()

    # Model training
    val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i_batch, batch_sample in enumerate(train_dataloader):
            if i_batch + 1 >= batches_per_epoch:
                break
            input = batch_sample['input'].to(device, non_blocking = True)
            output = batch_sample['output'].to(device, non_blocking = True)
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, output)
            loss.backward()
            optimizer.step()
            #running_loss += loss.item() * batch_sample['input'].size(0)
        #train_loss = running_loss / len(train_dataloader.dataset)
        #train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i_batch, batch_sample in enumerate(validation_dataloader):
                # Limiting validation data.
                if i_batch + 1 >= batches_per_epoch:
                    break
                input = batch_sample['input'].to(device, non_blocking = True)
                output = batch_sample['output'].to(device, non_blocking = True)
                outputs = model(input)
                loss = criterion(outputs, output)
                running_loss += loss.item() * input.size(0)
        val_loss = running_loss / np.min([len(validation_dataloader.dataset), batches_per_epoch*batch_size])
        #val_losses.append(val_loss)

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return val_loss

def tune_hyperparameters():
    os.makedirs("optuna_studies", exist_ok=True)
    study = optuna.create_study(direction='minimize', study_name='nn_control_alloc', storage="sqlite:///optuna_studies/nn_control_alloc_teste8.db", load_if_exists=True)
    study.optimize(objective, n_trials = 500)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics:')
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return study

if __name__ == '__main__':
    dataset_path = '../Datasets/Training datasets - v4/'
    train_dataloader, validation_dataloader, num_inputs = load_dataset(dataset_path)
    study = tune_hyperparameters()