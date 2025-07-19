from neural_network import NeuralNetwork, ControlAllocationDataset_Binary_Short
from dataset_handler import load_dataset
import torch
from torch.utils.data import DataLoader

from parameters.octorotor_parameters import num_rotors

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    optuna_version = 'v6_database_v5_smooth_kfold'
    datasets_folder = '../Datasets/Training datasets - v5/'
    nn_weights_path = datasets_folder + 'model_weights.pth'
    batch_size = 128
    pin_memory = True if device == 'cuda' else False

    test_dataset, num_inputs = load_dataset(datasets_folder, 'test_split_normalized.npy', num_rotors)
    test_dataset_class = ControlAllocationDataset_Binary_Short(test_dataset, num_rotors)
    test_dataloader = DataLoader(test_dataset_class, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=1)

    model = NeuralNetwork(optuna_version, num_inputs, num_rotors).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = getattr(torch.optim, model.optimizer)(model.parameters(), lr = model.opt_leaning_rate, weight_decay=model.l2_lambda)
    model.load_state_dict(torch.load(nn_weights_path, weights_only=True, map_location=device))

    # Test loop
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i_batch, batch_sample in enumerate(test_dataloader):
            input = batch_sample['input'].to(device, non_blocking = True)
            output = batch_sample['output'].to(device, non_blocking = True)
            outputs = model(input)
            loss = criterion(outputs, output)
            running_loss += loss.item() * input.size(0)
    test_loss = running_loss / len(test_dataloader.dataset)

    with open(datasets_folder + "test_loss.txt", "w") as f:
        f.write(str(test_loss))

    print('Test Loss =',test_loss)



