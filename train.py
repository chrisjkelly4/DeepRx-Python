import torch
import numpy as np
import model
import config
import torch_optimizer as toptim
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
import dataset
import time

deepRx_model = model.DeepRx(config.N_RX)

train_dataset = dataset.DeepRxDataset('/workspace/Datasets/full_training_data.h5')
train_loader = DataLoader(
    train_dataset,
    batch_size=80,
    shuffle=True,
    num_workers=4,
    worker_init_fn=dataset.worker_init_fn,
)

optimizer = toptim.Lamb(deepRx_model.parameters(), lr=1e-2, weight_decay=1e-4)

scheduler_warmup= LinearLR(optimizer, start_factor=1e-9, end_factor=1.0, total_iters=800)
scheduler_plateau = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=8200)
scheduler_decay = LinearLR(optimizer, start_factor=1.0, end_factor=1e-9, total_iters=21000)
scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_plateau, scheduler_decay],
    milestones=[800, 9000],
)

loss = torch.nn.BCEWithLogitsLoss()  # used for binary classification, and we are predicting bits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deepRx_model = deepRx_model.to(device)

pilot_mask = np.load('/workspace/pilot_mask.npy')


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
            prediction_masked = prediction[:, :, data_mask]
            prediction_flat = prediction_masked.permute(0, 2, 1).reshape(Z.shape[0], -1)

            loss_val = loss(prediction_flat, bits.float())
            loss_val.backward()
            optimizer.step()
            scheduler.step()

            iteration += 1
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                rate = iteration / elapsed
                remaining = (max_iterations - iteration) / rate
                print(
                        f"Iteration {iteration}/{max_iterations} | Loss: {loss_val.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | {rate:.1f} it/sec | ETA: {remaining / 3600:.1f} hrs")
            if iteration % 5000 == 0:
                torch.save(deepRx_model.state_dict(), f'/workspace/Checkpoints-ablation/deeprx_checkpoint_iteration{iteration}.pt')


if __name__ == '__main__':
    train_model(max_iterations=30000)

