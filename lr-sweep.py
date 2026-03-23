import torch
import numpy as np
import model
import config
import torch_optimizer as optim
from torch.utils.data import DataLoader
import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rates = [1e-2, 1e-3, 1e-4]
n_iterations = 2000

train_dataset = dataset.DeepRxDataset('/workspace/Datasets/full_training_data.h5')
train_loader = DataLoader(
    train_dataset,
    batch_size=80,
    shuffle=True,
    num_workers=4,
    worker_init_fn=dataset.worker_init_fn,
)

pilot_mask = np.load('pilot_mask.npy')
loss_fn = torch.nn.BCEWithLogitsLoss()

for lr in learning_rates:
    print(f"\n--- LR: {lr} ---")
    deepRx_model = model.DeepRx(config.N_RX).to(device)
    optimizer = optim.Lamb(deepRx_model.parameters(), lr=lr)

    iteration = 0
    for Z, bits in train_loader:
        if iteration >= n_iterations:
            break

        optimizer.zero_grad()
        Z, bits = Z.to(device), bits.to(device)
        prediction = deepRx_model.forward(Z)

        data_mask = ~pilot_mask.squeeze().astype(bool)
        prediction_masked = prediction[:, :, data_mask]
        prediction_flat = prediction_masked.reshape(Z.shape[0], -1)

        loss_val = loss_fn(prediction_flat, bits.float())
        loss_val.backward()
        optimizer.step()

        iteration += 1
        if iteration % 200 == 0:
            print(f"Iteration {iteration}/{n_iterations} | Loss: {loss_val.item():.4f}")