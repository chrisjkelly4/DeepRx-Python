import torch
import numpy as np
import model
import config
import utils
from torch.utils.data import DataLoader
import validation_dataset as dataset
import time

deepRx_model = model.DeepRx(config.N_RX)

val_dataset = dataset.DeepRxValDataset('/workspace/Datasets/full_validation_data.h5')
val_loader = DataLoader(
    val_dataset,
    batch_size=80,
    shuffle=False, #not valid to shuffle during validation, only aids in training
    num_workers=4,
    worker_init_fn=dataset.worker_init_fn,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deepRx_model.load_state_dict(torch.load('/workspace/Checkpoints/deeprx_checkpoint_iteration30000.pt'))
deepRx_model = deepRx_model.to(device)
deepRx_model.eval()
pilot_mask = np.load('/workspace/pilot_mask.npy')

def validate():
    print("Validation started.")
    all_bers = []
    all_snrs = []

    time_start = time.time()
    total_batches = len(val_loader)
    with torch.no_grad():
        for batch_idx, (Z, bits, snr) in enumerate(val_loader):
            Z = Z.to(device)

            bits = bits.to(device)

            prediction = deepRx_model.forward(Z)

            data_mask = ~pilot_mask.squeeze().astype(bool)
            prediction_masked = prediction[:, :, data_mask]
            prediction_flat = prediction_masked.permute(0, 2, 1).reshape(Z.shape[0], -1)

            predicted_bits = (prediction_flat > 0).float()

            # computer ber
            # Below is the general idea implimented as a for loop, but we are able to do there validation code
            # a little quicker using a some native numpy functions

            # for i in range (80):
            #     errors = (predicted_bits[i] != bits[i]).sum().item()
            #     total_bits = bits[i].numel()
            #     ber = errors / total_bits
            #
            #     all_bers.append(ber.item())
            #     all_snrs.append(snr.item())

            errors = (predicted_bits != bits)  # shape (80, 24576) - True where wrong
            bers = errors.sum(dim=1).float() / bits.shape[1]  # shape (80,) - one BER per sample

            all_bers.extend(bers.cpu().numpy().tolist())
            all_snrs.extend(snr.numpy().tolist())

            elapsed = time.time() - time_start
            rate = (batch_idx + 1) / elapsed  # batches per second
            remaining = (total_batches - (batch_idx + 1)) / rate
            print(f"Batch {batch_idx + 1}/{total_batches} | ETA: {remaining / 3600:.2f} hrs")

        group_centres, mean_bers = utils.group_ber_by_snr(
            np.array(all_snrs),
            np.array(all_bers)
        )

        np.save('/workspace/results/snr_group_centres.npy', group_centres)
        np.save('/workspace/results/mean_bers.npy', mean_bers)
        print("Validation complete, results saved.")

if __name__ == '__main__':
    validate()

