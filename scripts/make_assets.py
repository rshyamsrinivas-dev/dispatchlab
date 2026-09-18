"""Generate benchmark charts and a real simulator replay GIF, without a browser."""

import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "results" / "summary.json").read_text())
payload = json.loads((ROOT / "demo" / "replays.json").read_text())
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "figure.facecolor": "#f5f7f2",
        "axes.facecolor": "#f5f7f2",
        "text.color": "#122d37",
        "axes.labelcolor": "#122d37",
    }
)
colors = ["#aec7be", "#749e90", "#d2a36e", "#0b725d"]
labels = ["Nearest", "Earliest deadline", "Tuned batching", "Linear Q-learning"]
keys = ["nearest", "deadline", "batching", "linear_q_mean"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for ax, scenario, title in zip(
    axes, report["summary"], ["Standard demand", "Demand surge", "Tighter deadlines"]
):
    rows = report["summary"][scenario]
    vals = [100 * rows[k]["on_time_rate"] for k in keys]
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_title(title, loc="left", fontweight="bold", pad=15)
    ax.set_xlabel("On-time deliveries / all requests (%)")
    ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=10)
fig.suptitle(
    "DispatchLab / Held-out delivery performance",
    x=0.035,
    ha="left",
    fontweight="bold",
    fontsize=20,
)
fig.text(
    0.035,
    0.025,
    "300 unseen shifts per scenario · RL averages 3 training runs · Synthetic demand only · No real-world performance claim",
    fontsize=10,
    color="#577078",
)
fig.tight_layout(rect=[0.02, 0.075, 1, 0.92], w_pad=2.5)
fig.savefig(ROOT / "assets" / "benchmark.png", dpi=160)
plt.close(fig)

with (ROOT / "results" / "learning.csv").open() as f:
    history = list(csv.DictReader(f))
fig, ax = plt.subplots(figsize=(10, 4.7))
for seed, color in zip((11, 22, 33), ("#0b725d", "#c56c3b", "#527b9c")):
    rows = [r for r in history if int(r["training_seed"]) == seed]
    ax.plot(
        [int(r["episode"]) for r in rows],
        [float(r["validation_reward"]) for r in rows],
        color=color,
        label=f"Training seed {seed}",
    )
ax.axhline(
    report["batching_validation"][str(report["batching_threshold"])],
    color="#122d37",
    ls="--",
    label="Tuned batching · validation",
)
ax.set(
    title="Learning to dispatch: validation reward during training",
    xlabel="Training episodes",
    ylabel="Average validation reward",
)
ax.legend(frameon=False, ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "assets" / "learning.png", dpi=160)
plt.close(fig)

pts = np.array([[0, 0], [-2, 2], [2, 1], [0, -2]])
names = ["Depot", "North", "East", "South"]
travel = np.array([[0, 3, 2, 2], [3, 0, 4, 5], [2, 4, 0, 3], [2, 5, 3, 0]])
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
traces = [
    payload["replays"][f"standard|200001|{policy}"]["frames"]
    for policy in ("Learned policy", "Earliest deadline")
]


def frame(t):
    for ax, frames, title in zip(
        axes, traces, ("Learned dispatch policy", "Earliest-deadline rule")
    ):
        ax.clear()
        f = frames[t]
        for a in range(4):
            for b in range(a + 1, 4):
                ax.plot(
                    pts[[a, b], 0], pts[[a, b], 1], color="#d4ded9", ls=":", zorder=0
                )
        ax.scatter(
            pts[:, 0], pts[:, 1], s=400, color=["#122d37"] + ["#aec7be"] * 3, zorder=2
        )
        for i, (x, y) in enumerate(pts):
            ax.text(x, y - 0.4, names[i], ha="center", color="#577078", fontsize=11)
        pos = pts[f["location"]].astype(float)
        if f["travel_remaining"]:
            dest = pts[f["target"]]
            frac = 1 - f["travel_remaining"] / travel[f["location"], f["target"]]
            ax.plot(
                [pos[0], dest[0]], [pos[1], dest[1]], color="#0b725d", lw=3, zorder=1
            )
            pos += frac * (dest - pos)
        ax.scatter(
            *pos,
            marker="D",
            s=180,
            color="#0b725d",
            edgecolors="white",
            linewidths=2,
            zorder=3,
        )
        ax.set(xlim=(-3, 3), ylim=(-3.1, 3.1), aspect="equal")
        ax.axis("off")
        m = f["metrics"]
        ax.set_title(title, fontweight="bold", loc="left", fontsize=14)
        ax.text(
            0.02,
            0.98,
            f"Delivered {m['delivered']}/{m['requests']}   ·   On time {m['on_time']}   ·   Reward {m['reward']:.1f}",
            transform=ax.transAxes,
            fontsize=10,
        )
        cargo = sum(o["status"] for o in f["orders"])
        ax.text(
            0.02,
            0.03,
            f"Onboard {cargo}/3   ·   Depot queue {len(f['orders']) - cargo}",
            transform=ax.transAxes,
            fontsize=10,
        )
    fig.suptitle(
        f"DispatchLab / Same demand. Different decisions.     Tick {t:02}/48",
        x=0.055,
        ha="left",
        fontweight="bold",
        fontsize=17,
    )
    return []


fig.text(
    0.055,
    0.035,
    "Replay seed 200001 · Synthetic shift · Vehicle capacity 3 · Full benchmark covers 300 unseen shifts per scenario",
    fontsize=9,
    color="#577078",
)
fig.subplots_adjust(top=0.82, bottom=0.1, left=0.04, right=0.98, wspace=0.15)
frame(19)
fig.savefig(ROOT / "assets" / "preview.png", dpi=150)
anim = FuncAnimation(fig, frame, frames=49, interval=350)
anim.save(ROOT / "assets" / "dispatch.gif", writer=PillowWriter(fps=3), dpi=95)
plt.close(fig)
print("Wrote benchmark.png, learning.png, preview.png, dispatch.gif")
