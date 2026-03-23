import torch
import numpy as np
import model
import config
import torch_optimizer as optim
from torch.utils.data import DataLoader
import dataset
import time

deepRx_model = model.DeepRx(config.N_RX)

train_dataset = dataset.DeepRxDataset('workspace/Datasets/full_training_data.h5')
train_loader = DataLoader(
    train_dataset,
    batch_size=80,
    shuffle=True,
    num_workers=4,
    worker_init_fn=dataset.worker_init_fn,
)

optimizer = optim.Lamb(deepRx_model.parameters(), lr=1e-3)
loss = torch.nn.BCEWithLogitsLoss()  # used for binary classification, and we are predicting bits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deepRx_model = deepRx_model.to(device)

pilot_mask = np.load('pilot_mask.npy')


def train_model(max_iterations):
    iteration = 0
    start_time = time.time()
    while iteration < max_iterations:
        for Z, bits in train_loader:
            if iteration >= max_iterations:
                break

            optimizer.zero_grad()
            Z = Z.to(device)

            bits = bits.to(device)

            prediction = deepRx_model.forward(Z)

            # extract data positions only
            data_mask = ~pilot_mask.squeeze().astype(bool)  # True where DATA is flip to get data positions
            prediction_masked = prediction[:, :, data_mask]  # (10, 4, 6144)
            prediction_flat = prediction_masked.reshape(Z.shape[0], -1)  # (10, 24576)

            loss_val = loss(prediction_flat, bits.float())
            loss_val.backward()
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                rate = iteration / elapsed
                remaining = (max_iterations - iteration) / rate
                print(
                    f"Iteration {iteration}/{max_iterations} | Loss: {loss_val.item():.4f} | {rate:.1f} it/sec | ETA: {remaining / 3600:.1f} hrs")

        if iteration % 5000 == 0:
            torch.save(deepRx_model.state_dict(), f'deeprx_checkpoint_iteration{iteration}.pt')


if __name__ == '__main__':
    train_model(max_iterations=2)

##import h5py
##with h5py.File('../../../Desktop/training_data.h5', 'r') as f:
##    print(f['Y'].shape)
##    print(f['bits'].shape)
