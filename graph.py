import matplotlib.pyplot as plt
import numpy as np

sinr_db      = np.load('../results/snr_group_centres.npy')
baseline_ber = np.load('../results/mean_bers.npy')
ablation_ber = np.load('../results-ablation/ablation_mean_bers.npy')

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(sinr_db, baseline_ber, 'b-o',  linewidth=2, markersize=6, label="DeepRx (baseline)")
ax.semilogy(sinr_db, ablation_ber, 'r--s', linewidth=2, markersize=6, label="DeepRx (ablation – no Hr)")

ax.set_xlabel("SINR (dB)", fontsize=12)
ax.set_ylabel("Uncoded BER", fontsize=12)
ax.set_title("Uncoded BER", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("../ber_comparison.png", dpi=150, bbox_inches="tight")
plt.show()