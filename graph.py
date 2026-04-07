import matplotlib.pyplot as plt
import numpy as np

# ── data ──────────────────────────────────────────────────────────────────────
# Replace these with your actual results
sinr_db = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])

baseline_ber  = np.array([0.20, 0.09, 0.04, 0.015, 0.004, 0.0015, 0.0005, 0.00018, 0.00012])
ablation_ber  = np.array([0.22, 0.13, 0.07, 0.04,  0.025, 0.018,  0.013,  0.010,   0.009])

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

titles      = ["Uncoded BER", "Coded BER"]
y_labels    = ["Uncoded BER", "Coded BER"]

for ax, title, y_label in zip(axes, titles, y_labels):
    ax.semilogy(sinr_db, baseline_ber, 'b-o',  linewidth=2, markersize=6, label="DeepRx (baseline)")
    ax.semilogy(sinr_db, ablation_ber, 'r--s', linewidth=2, markersize=6, label="DeepRx (ablation – no Hr)")

    ax.set_xlabel("SINR (dB)", fontsize=12)
    ax.set_ylabel(y_label,     fontsize=12)
    ax.set_title(title,        fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_ylim([1e-4, 1])

plt.tight_layout()
plt.savefig("ber_comparison.png", dpi=150, bbox_inches="tight")
plt.show()