import matplotlib.pyplot as plt
import numpy as np

sinr_db      = np.load('../results/snr_group_centres.npy')
baseline_ber = np.load('../results/mean_bers.npy')
ablation_ber = np.load('../results-ablation/ablation_mean_bers.npy')

mean_baseline_ber = np.mean(baseline_ber, axis=0)
mean_ablation_ber = np.mean(ablation_ber, axis=0)

initial_gap = ablation_ber[0] - baseline_ber[0]
final_gap = ablation_ber[-1] - baseline_ber[-1]

raw_gaps = ablation_ber - baseline_ber
pct_gaps = (ablation_ber - baseline_ber) / baseline_ber * 100

max_raw_idx = np.argmax(raw_gaps)
max_pct_idx = np.argmax(pct_gaps)

print(f"Widest raw gap: {raw_gaps[max_raw_idx]:.4f} at SINR {sinr_db[max_raw_idx]:.1f} dB")
print(f"Widest percentage gap: {pct_gaps[max_raw_idx]:.2f}% at SINR {sinr_db[max_raw_idx]:.1f} dB")

print (f"Initial raw difference: {initial_gap}")
print (f"Initial percentage difference: {initial_gap/baseline_ber[0]*100}% ")
print (f"Final raw difference: {final_gap}")
print (f"Final percentage difference: {final_gap/baseline_ber[-1]*100}% ")
print(f"Raw difference: {(mean_ablation_ber-mean_baseline_ber)}")
print(f"Percetenatge difference: {(mean_ablation_ber-mean_baseline_ber)/mean_baseline_ber * 100}% ")

fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(sinr_db, baseline_ber, 'b-o',  linewidth=2, markersize=6, label="DeepRx (baseline)")
ax.semilogy(sinr_db, ablation_ber, 'r--s', linewidth=2, markersize=6, label="DeepRx (ablation – no H)")

ax.set_xlabel("SINR (dB)", fontsize=12)
ax.set_ylabel("Uncoded BER", fontsize=12)
ax.set_title("Uncoded BER", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("../ber_comparison.png", dpi=150, bbox_inches="tight")
plt.show()