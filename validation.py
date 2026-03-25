import torch
import numpy as np
import model
import config
from torch.utils.data import DataLoader
import dataset
import time

deepRx_model = model.DeepRx(config.N_RX)

val_dataset = dataset.DeepRxDataset('/workspace/Datasets/full_validation_data.h5')
val_loader = DataLoader(
    val_dataset,
    batch_size=80,
    shuffle=True,
    num_workers=4,
    worker_init_fn=dataset.worker_init_fn,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deepRx_model = deepRx_model.to(device)

pilot_mask = np.load('/workspace/pilot_mask.npy')

def validate():
    deepRx_model.eval()
