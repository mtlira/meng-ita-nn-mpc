import numpy as np
from neural_network import NeuralNetwork, ControlAllocationDataset_Binary, EarlyStopper, ControlAllocationDataset_Binary_Short
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import optuna
from optuna.trial import TrialState
from torch import nn
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from parameters.octorotor_parameters import num_rotors

# Hyperparameters
num_epochs = 35
batch_size = 128
#learning_rate = 0.001
num_outputs = num_rotors
num_neurons_hiddenlayers = 128
batches_per_epoch = 300
k_folds = 5

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#device = torch.device('cpu')
print(f"Using {device} device")

# Dataset path
def load_dataset(datasets_folder):
    #global_dataset = ControlAllocationDataset_Binary(datasets_folder, False, num_outputs)
    #train_size = int(0.9 * len(global_dataset))
    #test_size = len(global_dataset) - train_size
    #val_size = int(0.2 * len(global_dataset))
    
    #train_dataset, test_dataset = torch.utils.data.random_split(global_dataset, [train_size, test_size])
    #num_inputs = global_dataset.num_inputs

    train_dataset = np.load(datasets_folder + 'training_split_normalized.npy')
    num_inputs = len(train_dataset[0]) - num_outputs
    print('shape dataset',np.shape(train_dataset))
    print('number of inputs',num_inputs)
    return train_dataset, num_inputs

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
    global k_folds

    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float('optimizer_learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('l2_lambda', 0, 0.5)

    # Loss function
    criterion = torch.nn.MSELoss()

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=50)
    val_losses = []

    earlystopper = EarlyStopper(4, 12, 0.05)

    for train_idx, val_idx in kf.split(train_dataset):
        train_subset = Subset(train_dataset_class, train_idx)
        val_subset = Subset(train_dataset_class, val_idx)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Generate the model
        model = define_model(trial, num_inputs).to(device)

        # Generate the optimizers
        optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

        # Model training
        val_loss = np.inf
        best_val_loss = np.inf # Best validation loss of each fold configuration
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i_batch, batch_sample in enumerate(train_dataloader):
                if i_batch + 1 >= batches_per_epoch:
                    break
                optimizer.zero_grad()
                input = batch_sample['input'].to(device, non_blocking = True)
                output = batch_sample['output'].to(device, non_blocking = True)
                predicted_output = model(input)
                loss = criterion(predicted_output, output)
                loss.backward()
                optimizer.step()
                #running_loss += loss.item() * batch_sample['input'].size(0)
            #train_loss = running_loss / len(train_dataloader.dataset)
            #train_losses.append(train_loss)

            # Validation loop
            model.eval()
            running_loss = 0.0
            num_samples = 0
            with torch.no_grad():
                for i_batch, batch_sample in enumerate(validation_dataloader):
                    # Limiting validation data.
                    input = batch_sample['input'].to(device, non_blocking = True)
                    output = batch_sample['output'].to(device, non_blocking = True)
                    predicted_output = model(input)
                    loss = criterion(predicted_output, output)
                    running_loss += loss.item() * input.size(0)
                    num_samples += input.size(0)
            val_loss = running_loss / num_samples

            # Report intermediate value to optuna
            trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            best_val_loss = min(best_val_loss, val_loss)
            
            if earlystopper.stop_early(best_val_loss, model): break   
        
        # Finished all the epochs of a specific fold combination --> Saving the best validation loss of the fold configutation
        val_losses.append(best_val_loss)
    return np.mean(val_losses)

def tune_hyperparameters():
    os.makedirs("optuna_studies", exist_ok=True)
    study = optuna.create_study(direction='minimize', study_name='nn_control_alloc', storage="sqlite:///optuna_studies/nn_control_alloc_v6_smooth.db", load_if_exists=True)
    study.optimize(objective, n_trials = 120)

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
    dataset_path = '../Datasets/Training datasets - v5/'
    train_dataset, num_inputs = load_dataset(dataset_path)
    train_dataset_class = ControlAllocationDataset_Binary_Short(train_dataset, num_outputs)
    study = tune_hyperparameters()