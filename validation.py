import torch
import numpy as np
import model
import config
from torch.utils.data import DataLoader
import validation_dataset as dataset
import time

deepRx_model = model.DeepRx(config.N_RX)

val_dataset = dataset.DeepRxValDataset('/workspace/Datasets/full_validation_data.h5')
val_loader = DataLoader(
    val_dataset,
    batch_size=80,
    shuffle=True,
    num_workers=4,
    worker_init_fn=dataset.worker_init_fn,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deepRx_model = deepRx_model.to(device)
deepRx_model.eval()
deepRx_model.load_state_dict(torch.load('/workspace/Checkpoints/deeprx_checkpoint_iteration30000.pt'))
pilot_mask = np.load('/workspace/pilot_mask.npy')

def validate():
    for Z, bits, snr in val_loader:
        with torch.no_grad():
            torch.no_grad()
            Z = Z.to(device)

            bits = bits.to(device)

            prediction = deepRx_model.forward(Z)

            data_mask = ~pilot_mask.squeeze().astype(bool)
            prediction_masked = prediction[:, :, data_mask]
            prediction_flat = prediction_masked.permute(0, 2, 1).reshape(Z.shape[0], -1)

            prediction_threshhold = torch.max(prediction_flat, dim=1)[0]
            # computer ber

