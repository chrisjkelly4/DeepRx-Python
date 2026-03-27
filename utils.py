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
