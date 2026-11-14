import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Set aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

# Data definition based on experimental benchmarks & records
models = [
    "Baseline TCCFNet v1\n(ResNet-50 + CLS)",
    "Hybrid TCCFNet v2\n(ResNet-50 + Skips)",
    "SegFormer (MiT-B3)\n(Hierarchical ViT + MLP)",
]

metrics = ["IoU (Jaccard)", "Dice / F1", "Precision", "Recall", "Accuracy"]

# Scores for each model
scores = {
    "Baseline TCCFNet v1\n(ResNet-50 + CLS)": [0.4716, 0.6505, 0.5192, 0.8706, 0.5710],
    "Hybrid TCCFNet v2\n(ResNet-50 + Skips)": [0.3457, 0.4415, 0.7222, 0.5930, 0.5950],
    "SegFormer (MiT-B3)\n(Hierarchical ViT + MLP)": [0.8740, 0.9320, 0.9140, 0.9510, 0.9480],
}

colors = ["#e74c3c", "#f39c12", "#27ae60"]  # Red, Amber, Emerald Green

# Create 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(
    "MangroveNet Architecture Performance Comparison:\nSegFormer (MiT-B3) vs TCCFNet Baselines",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# -----------------------------------------------------------------------------
# 1. Grouped Bar Chart across all core metrics
# -----------------------------------------------------------------------------
ax1 = axes[0, 0]
x = np.arange(len(metrics))
bar_width = 0.25

for i, (model_name, vals) in enumerate(scores.items()):
    offset = (i - 1) * bar_width
    bars = ax1.bar(
        x + offset,
        vals,
        bar_width,
        label=model_name.replace("\n", " "),
        color=colors[i],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9,
    )
    for bar in bars:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

ax1.set_title("A. Core Segmentation Metrics Comparison", fontsize=12, fontweight="bold", pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontweight="semibold")
ax1.set_ylabel("Metric Score (0.0 to 1.0)", fontweight="semibold")
ax1.set_ylim(0, 1.15)
ax1.legend(loc="upper left", frameon=True, fontsize=9)
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# -----------------------------------------------------------------------------
# 2. Focus on Primary Overlap Metrics (IoU & Dice) with Gain Callouts
# -----------------------------------------------------------------------------
ax2 = axes[0, 1]
primary_metrics = ["IoU (Jaccard Index)", "Dice Coefficient (F1)"]
tccf_v1_prim = [scores[models[0]][0], scores[models[0]][1]]
tccf_v2_prim = [scores[models[1]][0], scores[models[1]][1]]
segformer_prim = [scores[models[2]][0], scores[models[2]][1]]

x2 = np.arange(len(primary_metrics))
w2 = 0.24

ax2.bar(x2 - w2, tccf_v1_prim, w2, label="TCCFNet v1", color=colors[0], edgecolor="black", alpha=0.9)
ax2.bar(x2, tccf_v2_prim, w2, label="TCCFNet v2", color=colors[1], edgecolor="black", alpha=0.9)
b3 = ax2.bar(x2 + w2, segformer_prim, w2, label="SegFormer (MiT-B3)", color=colors[2], edgecolor="black", alpha=0.95)

# Annotate the dramatic IoU gain
iou_gain = segformer_prim[0] - tccf_v1_prim[0]
dice_gain = segformer_prim[1] - tccf_v1_prim[1]

ax2.annotate(
    f"+{iou_gain * 100:.1f}% IoU Gain\n(0.47 -> 0.87)",
    xy=(x2[0] + w2, segformer_prim[0]),
    xytext=(x2[0] + w2 - 0.15, 1.02),
    arrowprops=dict(facecolor="#27ae60", arrowstyle="->", lw=2),
    fontsize=9.5,
    fontweight="bold",
    color="#1e8449",
    ha="center",
)

ax2.annotate(
    f"+{dice_gain * 100:.1f}% Dice Gain\n(0.65 -> 0.93)",
    xy=(x2[1] + w2, segformer_prim[1]),
    xytext=(x2[1] + w2 - 0.15, 1.02),
    arrowprops=dict(facecolor="#27ae60", arrowstyle="->", lw=2),
    fontsize=9.5,
    fontweight="bold",
    color="#1e8449",
    ha="center",
)

for idx, val in enumerate(segformer_prim):
    ax2.text(x2[idx] + w2, val / 2, f"{val:.4f}", ha="center", va="center", color="white", fontweight="bold")

for idx, val in enumerate(tccf_v1_prim):
    ax2.text(x2[idx] - w2, val / 2, f"{val:.4f}", ha="center", va="center", color="white", fontweight="bold")

for idx, val in enumerate(tccf_v2_prim):
    ax2.text(x2[idx], val / 2, f"{val:.4f}", ha="center", va="center", color="white", fontweight="bold")

ax2.set_title("B. Target Overlap Metric Leap (IoU & Dice)", fontsize=12, fontweight="bold", pad=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(primary_metrics, fontweight="semibold")
ax2.set_ylabel("Score", fontweight="semibold")
ax2.set_ylim(0, 1.2)
ax2.legend(loc="upper right", frameon=True, fontsize=9)
ax2.grid(axis="y", linestyle="--", alpha=0.5)

# -----------------------------------------------------------------------------
# 3. Radar / Spider Chart of Model Profiles
# -----------------------------------------------------------------------------
ax3 = plt.subplot(2, 2, 3, polar=True)
radar_metrics = ["IoU", "Dice", "Precision", "Recall", "Accuracy"]
num_vars = len(radar_metrics)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close loop

for i, (model_name, vals) in enumerate(scores.items()):
    v_closed = vals + vals[:1]
    short_label = model_name.split("\n")[0]
    ax3.plot(angles, v_closed, color=colors[i], linewidth=2.2, label=short_label)
    ax3.fill(angles, v_closed, color=colors[i], alpha=0.15)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(radar_metrics, fontweight="bold", fontsize=10)
ax3.set_ylim(0, 1.0)
ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#555")
ax3.set_title("C. Multi-Dimensional Performance Profile", fontsize=12, fontweight="bold", pad=20)
ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

# -----------------------------------------------------------------------------
# 4. Precision vs. Recall Trade-Off Space
# -----------------------------------------------------------------------------
ax4 = axes[1, 1]

for i, (model_name, vals) in enumerate(scores.items()):
    prec = vals[2]
    rec = vals[3]
    iou = vals[0]
    short_name = model_name.split("\n")[0]

    # Bubble size proportional to IoU
    ax4.scatter(
        rec,
        prec,
        s=iou * 1800,
        color=colors[i],
        alpha=0.75,
        edgecolors="black",
        linewidth=1.5,
        label=f"{short_name} (IoU: {iou:.2f})",
    )
    ax4.annotate(
        f"{short_name}\nP={prec:.2f}, R={rec:.2f}",
        xy=(rec, prec),
        xytext=(rec - 0.04, prec + 0.04),
        fontsize=9.5,
        fontweight="bold",
        color="#2c3e50",
    )

# Ideal point
ax4.scatter(1.0, 1.0, color="#8e44ad", marker="*", s=300, label="Ideal Classifier (1.0, 1.0)")
ax4.annotate("Ideal Point\n(P=1.0, R=1.0)", xy=(1.0, 1.0), xytext=(0.88, 1.03), fontsize=9, fontweight="bold", color="#8e44ad")

ax4.set_title("D. Precision vs. Recall Operational Space\n(Bubble Size = IoU)", fontsize=12, fontweight="bold", pad=12)
ax4.set_xlabel("Recall (Sensitivity)", fontweight="semibold")
ax4.set_ylabel("Precision", fontweight="semibold")
ax4.set_xlim(0.5, 1.05)
ax4.set_ylim(0.4, 1.1)
ax4.grid(True, linestyle="--", alpha=0.5)
ax4.legend(loc="lower left", frameon=True, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])

output_filename = "segformer_performance_comparison.png"
plt.savefig(output_filename, dpi=300)
plt.close()

print(f"✅ Performance comparison graph successfully saved to: {output_filename}")
