import tensorflow as tf
import numpy as np
import torch

## Old version
##def construct_input_tensor(Y, Xp):
##    # Re-Size input tensors
##    Hr = Y * np.conj(Xp)
##    Y = tf.squeeze(Y, axis=[0, 1])
##    Xp = tf.expand_dims(Xp, axis=0)
##    Hr = tf.squeeze(Hr, axis=[0, 1])
##
##    # Combine input Tenosrs into one Input
##    Z = tf.concat([Y, Xp, Hr], axis=0)
##    Z = tf.concat([tf.math.imag(Z), tf.math.real(Z)], axis=0)
##
##    # Add batch dimension for batch training
##    # Z = tf.expand_dims(Z, axis=0)  # → (1, 10, 14, 512)
##
##
##    # Convert Input Tensor from Tensor Flow to PyTorch
##    Z = tf.cast(Z, tf.float32)  # ensure float32
##    Z = torch.tensor(Z.numpy())  # TF → PyTorch tensor
##
##    return Z


def construct_input_tensor(Y, Xp):
    # Compute raw channel estimate
    Hr = Y * torch.conj(Xp)
    
    # Re-size input tensors
    Y = Y.squeeze(0).squeeze(0)    # remove batch and tx dims
    Xp = Xp.unsqueeze(0)           # add rx dim
    Hr = Hr.squeeze(0).squeeze(0)  # remove batch and tx dims
    
    # Combine into one input tensor
    Z = torch.cat([Y, Xp, Hr], dim=0)
    
    # Split real and imaginary parts
    Z = torch.cat([Z.real, Z.imag], dim=0)
    
    return Z.float()
