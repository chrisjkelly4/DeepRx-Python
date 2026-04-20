# DeepRx Python

A Python reimplementation of **DeepRx**, a fully convolutional deep learning OFDM receiver proposed by Honkala, Korpi and Huttunen at Nokia Bell Labs (2021). The original paper implemented DeepRx in MATLAB; this project makes it open source and accessible in Python using PyTorch and Sionna, and extends the work with an ablation study quantifying the impact of the raw channel estimate on BER performance.

This implementation was produced as part of a BSc Computer Science dissertation at Loughborough University.

> Honkala, M., Korpi, D. and Huttunen, J.M.J. (2021) 'DeepRx: Fully convolutional deep learning receiver', *IEEE Transactions on Wireless Communications*, 20(6), pp. 3925–3940. doi: 10.1109/TWC.2021.3054520

---

## What is DeepRx?

DeepRx replaces the traditional OFDM receiver pipeline with a fully convolutional neural network, trained end-to-end as a supervised learning problem. Instead of separate channel estimation, equalisation and demapping stages, DeepRx takes a frequency-domain OFDM resource grid as input and outputs Log-Likelihood Ratios (LLRs) directly, one per bit per resource element. These LLRs can be interpreted as bit predictions with associated confidence, or fed into a 5G-compliant LDPC decoder.

The input tensor Z is constructed from three components stacked along the channel dimension:

- **Y** — the raw received signal as seen by the receiver antennas
- **Xp** — the known pilot reference symbols (zeros at non-pilot positions)  
- **Hr** — a raw channel estimate computed as the element-wise product of Y and the complex conjugate of Xp at pilot positions

---

## Ablation Study

The original paper states that including Hr "allows for a somewhat easier and faster learning process" but provides no quantitative evidence of its impact on BER. This project directly tests that claim.

**Hypothesis:** Providing DeepRx with the explicit raw channel estimate Hr improves BER performance beyond simply accelerating convergence.

**Method:** Two models were trained under identical conditions. The baseline model receives the full input tensor (Y, Xp, Hr). The ablation model receives only (Y, Xp), with the input channel count reduced accordingly.

**Results:**

| Model | BER @ -4dB | BER @ 20dB |
|---|---|---|
| Baseline (with Hr) | 0.444 | 0.347 |
| Ablation (without Hr) | 0.466 | 0.370 |

The ablation model performs approximately 8.3% worse on average across all SINR values. The gap widens at high SINR, suggesting Hr contributes meaningful information beyond convergence speed. Whether this reflects a fundamental dependency or insufficient training iterations for the ablation model remains an open question for future work.

---

## Project Structure

```
DeepRx-Python/
├── config.py       # All constants and hyperparameters
├── model.py        # ResNetBlock and DeepRx architecture classes
├── dataset.py      # Sionna data generation and PyTorch Dataset class
├── train.py        # Training loop
├── utils.py        # Input tensor construction and BER evaluation helpers
├── evaluate.py     # Validation and BER calculation
```

---

## Architecture

The network follows Table 1 of the original paper:

- 11 preactivation ResNet blocks with depthwise separable convolutions
- Two additional 2D convolution layers: one input projection, one output projection
- Dilated convolutions matching the paper's dilation schedule
- Channel widths: 64 → 64 → 128 → 128 → 256 → 256 → 256 → 128 → 128 → 64 → 64
- Input channels: 2 × (2 × N_RX + 1) = 10 with N_RX = 2 (real and imaginary parts stacked separately)
- Output: 4 LLRs per resource element (16-QAM)

The ablation model uses the same architecture with input channels reduced to 2 × (N_RX + 1) = 6, removing the Hr component.

---

## Data Generation

Data generation uses **Sionna 2.0** to simulate 5G NR PUSCH OFDM transmissions with 3GPP channel models.

| Parameter | Value |
|---|---|
| Carrier frequency | 4 GHz |
| Number of PRBs | 26 (312 subcarriers) |
| Subcarrier spacing | 15 kHz |
| OFDM symbols per TTI | 14 |
| Modulation scheme | 16-QAM |
| RX antennas | 2 |
| TX antennas | 1 |
| FFT size | 512 |
| Training TTIs | 300,000 |
| Validation TTIs | 200,000 |

**Training channel models:** CDL-B, CDL-C, CDL-D, TDL-B, TDL-C, TDL-D

**Validation channel models:** CDL-A, CDL-E, TDL-A, TDL-E

Parameters randomised per sample: SNR (-4 to 32 dB), Doppler shift (0–500 Hz), RMS delay spread (10–300 ns).

Y and Xp are saved to HDF5 files. Hr is computed at training time from Y and Xp to keep the ablation straightforward to implement.

---

## Training

- **Optimiser:** LAMB (layer-wise adaptive moments), lr = 1e-2
- **Batch size:** 80 TTIs
- **Iterations:** 30,000
- **Loss:** BCEWithLogitsLoss on data positions only (pilot positions masked)
- **Hardware:** NVIDIA RTX 6000 Ada (single GPU)
- **Checkpoints:** Saved every 5000 iterations

Note: The original paper trained on four NVIDIA 2080Ti GPUs in parallel with an effective batch size of 320. The reduced batch size used here likely contributes to the performance gap between this implementation and the paper's reported results (BER < 0.01 at high SINR vs approximately 0.35 here).

---

## Dependencies

```
torch
sionna>=2.0
tensorflow        # required by Sionna for data generation
h5py
torch-optimizer   # for LAMB optimiser
numpy
matplotlib
```

> **Note:** Sionna requires a Linux environment or Google Colab. It is not compatible with Apple Silicon (M1/M2) Macs.

---

## Known Limitations

- **No LDPC coding:** Only uncoded BER is evaluated. The paper reports both uncoded and coded BER.
- **No interference:** SIR is sampled during data generation but not applied to the signal.
- **SIMO only:** N_TX = 1, N_RX = 2. MIMO extension is noted as future work in the original paper.
- **Compute scale:** Single GPU training with batch size 80 versus the paper's four-GPU setup with effective batch size 320.

---

## Citation

If you use this code, please cite the original DeepRx paper:

```
@article{honkala2021deeprx,
  title={DeepRx: Fully convolutional deep learning receiver},
  author={Honkala, Mikko and Korpi, Dani and Huttunen, Janne M J},
  journal={IEEE Transactions on Wireless Communications},
  volume={20},
  number={6},
  pages={3925--3940},
  year={2021},
  doi={10.1109/TWC.2021.3054520}
}
```