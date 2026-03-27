import numpy as np
import torch

def construct_input_tensor(Y, Xp):
    # Compute raw channel estimate
    Hr = Y * torch.conj(Xp)
    
    # Re-size input tensors
    Y = Y.squeeze(0).squeeze(0)    # remove batch and tx dims
    Xp = Xp.unsqueeze(0)           # add rx dim

    # Removed Hr Components for ablation study
    Hr = Hr.squeeze(0).squeeze(0)  # remove batch and tx dims
    
    # Combine into one input tensor
    Z = torch.cat([Y, Xp, Hr], dim=0)

    # Split real and imaginary parts
    Z = torch.cat([Z.real, Z.imag], dim=0)

    return Z.float()


def construct_input_tensor_ablation(Y, Xp):
    # Re-size input tensors
    Y = Y.squeeze(0).squeeze(0)  # remove batch and tx dims
    Xp = Xp.unsqueeze(0)  # add rx dim

    # Combine into one input tensor
    Z = torch.cat([Y, Xp], dim=0)

    # Split real and imaginary parts
    Z = torch.cat([Z.real, Z.imag], dim=0)

    return Z.float()

def group_ber_by_snr(ber_array, snr_array):
    NUM_OF_EDGES = 26
    mean_ber = []
    group_centers = np.arange(-4, 21, 1)

    group_edges = np.linspace(-4.5, 20.5, NUM_OF_EDGES)
    grouped_snr = np.digitize(snr_array, group_edges)
    for i in range (1,NUM_OF_EDGES):
        mask = grouped_snr == i
        if mask.sum() > 0:
            mean_ber.append(ber_array[mask].mean())
        else:
            mean_ber.append(np.nan)

    return group_centers, mean_ber
