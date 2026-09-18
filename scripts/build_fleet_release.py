"""Build the offline fleet replay, figures, animation and measured report."""

from pathlib import Path
import json, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "advanced"
data = json.loads((base / "demo/replays.json").read_text())
report = data["report"]
template = (base / "demo/template.html").read_text()
(base / "demo/index.html").write_text(
    template.replace("__PAYLOAD__", json.dumps(data, separators=(",", ":")))
)
assets = base / "assets"
assets.mkdir(exist_ok=True)
dark = "#0b121b"
ink = "#edf4f7"
muted = "#91a7b8"
mint = "#b4f273"
teal = "#56d4c0"
orange = "#ffb36d"
plt.rcParams.update(
    {
        "figure.facecolor": dark,
        "axes.facecolor": dark,
        "text.color": ink,
        "axes.labelcolor": muted,
        "xtick.color": muted,
        "ytick.color": muted,
        "axes.edgecolor": "#263747",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)
keys = ["nearest", "deadline", "planner", "ddqn_mean"]
labels = ["Nearest", "Earliest deadline", "Route planner", "Double DQN"]
titles = ["Standard", "Demand surge", "Traffic disruption", "Limited battery"]
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
for ax, (scenario, rows), title in zip(axs, report["summary"].items(), titles):
    vals = [100 * rows[k]["on_time_rate"] for k in keys]
    bars = ax.barh(labels, vals, color=["#4b667a", "#7793a1", teal, mint], height=0.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_title(title, loc="left", fontweight="bold", pad=16)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, color=ink, fontsize=9)
    ax.set_xlabel("On-time / all requests (%)")
fig.suptitle(
    "DispatchLab Fleet / What the held-out shifts show",
    x=0.025,
    ha="left",
    fontsize=20,
    fontweight="bold",
)
fig.text(
    0.025,
    0.035,
    f"{report['test_days']} unseen shifts per condition · Neural policy averages 3 training runs · Synthetic data only",
    color=muted,
    fontsize=10,
)
fig.tight_layout(rect=[0, 0.1, 1, 0.91], w_pad=2)
fig.savefig(assets / "benchmark.png", dpi=150)
plt.close(fig)
history = list(csv.DictReader((base / "results/learning.csv").open()))
fig, ax = plt.subplots(figsize=(11, 4.7))
for seed, color in zip(report["replicas"], [teal, orange, mint]):
    rows = [r for r in history if int(r["seed"]) == seed]
    ax.plot(
        [int(r["episode"]) for r in rows],
        [float(r["validation_reward"]) for r in rows],
        color=color,
        label=f"Training seed {seed}",
    )
planner = report["planner_validation"][str(report["planner_batch"])]
ax.axhline(planner, color=ink, ls="--", label="Planner validation reward")
ax.axvline(1500, color=muted, ls=":", alpha=0.7)
ax.set(
    xlabel="Training episodes",
    ylabel="Validation reward",
    title="Learning curves, including the lower-rate refinement stage",
)
ax.legend(frameon=False, labelcolor=ink, ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(assets / "learning.png", dpi=150)
plt.close(fig)

font = font_manager.findfont("DejaVu Sans")
bold = font_manager.findfont(
    font_manager.FontProperties(family="DejaVu Sans", weight="bold")
)


def ft(size, weight=False):
    return ImageFont.truetype(bold if weight else font, size)


def make_frame(t):
    im = Image.new("RGB", (1400, 850), dark)
    d = ImageDraw.Draw(im)
    d.text((38, 24), "DispatchLab  /  Fleet", font=ft(29, True), fill=ink)
    d.text((38, 74), "One fleet. Two ways to dispatch.", font=ft(37, True), fill=ink)
    d.text(
        (38, 124),
        "Two vehicles · Battery constraints · Uncertain travel · Delivery deadlines",
        font=ft(17),
        fill=muted,
    )
    d.text((1190, 35), f"TICK {t:02} / 60", font=ft(17, True), fill=mint)
    for panel, policy in enumerate(["Double DQN", "Route planner"]):
        left = 38 + panel * 674
        right = left + 650
        f = data["replays"][f"standard|2000001|{policy}"]["frames"][t]
        m = f["metrics"]
        d.rounded_rectangle(
            (left, 176, right, 782),
            radius=16,
            fill="#111d29",
            outline="#263747",
            width=2,
        )
        d.text(
            (left + 22, 192),
            policy,
            font=ft(23, True),
            fill=mint if panel == 0 else teal,
        )
        d.text(
            (left + 22, 232),
            f"{m['delivered']}/{m['requests']} delivered     {m['on_time']} on time     {m['reward']:.1f} reward",
            font=ft(16),
            fill=ink,
        )
        pts = [
            (left + 325, 460),
            (left + 115, 340),
            (left + 325, 340),
            (left + 535, 340),
            (left + 115, 580),
            (left + 325, 580),
            (left + 535, 580),
        ]
        for x in [left + 115, left + 220, left + 325, left + 430, left + 535]:
            d.line((x, 292, x, 629), fill="#263b4b", width=13)
        for y in [340, 460, 580]:
            d.line((left + 62, y, right - 60, y), fill="#263b4b", width=13)
        for i, v in enumerate(f["vehicles"]):
            if v["remaining"]:
                x, y = pts[v["location"]]
                tx, ty = pts[v["target"]]
                d.line([(x, y), (tx, y), (tx, ty)], fill=[teal, orange][i], width=5)
        for i, (x, y) in enumerate(pts):
            color = mint if i == 0 else "#39556a"
            d.ellipse(
                (x - 13, y - 13, x + 13, y + 13), fill=color, outline=ink, width=1
            )
            name = data["nodes"][i]
            box = d.textbbox((0, 0), name, font=ft(14))
            d.text((x - (box[2] - box[0]) / 2, y + 20), name, font=ft(14), fill=ink)
        for i, v in enumerate(f["vehicles"]):
            x, y = pts[v["location"]]
            if v["remaining"]:
                tx, ty = pts[v["target"]]
                distance = abs(tx - x) + abs(ty - y)
                p = (1 - v["remaining"] / v["duration"]) * distance
                dx = abs(tx - x)
                if p <= dx:
                    x += np.sign(tx - x) * p
                else:
                    x = tx
                    y += np.sign(ty - y) * (p - dx)
            else:
                x += -18 if i == 0 else 18
                y -= 18
            color = [teal, orange][i]
            d.polygon([(x, y - 10), (x + 10, y), (x, y + 10), (x - 10, y)], fill=color)
            d.text((x + 14, y - 16), f"V{i + 1}", font=ft(13, True), fill=color)
            cx = left + 22 + i * 312
            cy = 670
            cargo = sum(o["vehicle"] == i for o in f["orders"])
            d.text(
                (cx, cy),
                f"V{i + 1}  {cargo}/3 parcels   {v['battery']}/{f['battery_capacity']} energy",
                font=ft(14),
                fill=color,
            )
            d.rounded_rectangle(
                (cx, cy + 30, cx + 280, cy + 36), radius=3, fill="#304354"
            )
            width = 280 * v["battery"] / f["battery_capacity"]
            if width > 0:
                d.rounded_rectangle(
                    (cx, cy + 30, cx + width, cy + 36), radius=3, fill=color
                )
        d.text(
            (left + 22, 740),
            f"{len(f['orders'])} active orders · {m['rejected']} rejected",
            font=ft(14),
            fill=muted,
        )
    d.text(
        (38, 810),
        "Actual simulator replay · Fixed example seed 2000001 · Full benchmark and source included",
        font=ft(14),
        fill=muted,
    )
    return im


make_frame(24).save(assets / "preview.png")
frames = [make_frame(t) for t in range(0, 61, 2)]
frames[0].save(
    assets / "fleet.gif",
    save_all=True,
    append_images=frames[1:],
    duration=300,
    loop=0,
    optimize=False,
)

lines = [
    "# What the fleet benchmark showed",
    "",
    "I kept the route planner in the comparison because a logistics project needs a credible operational baseline. The neural model has to justify itself on measured outcomes.",
    "",
    "## Experiment setup",
    "",
    "- Three neural replicas: seeds 101, 202, 303. Each trained for 1,500 episodes, then 2,000 refinement episodes on fresh training scenarios.",
    "- Stage-one demand seeds: 300000–301499, 350000–351499, 400000–401499. Refinement: 600000–601999, 650000–651999, 700000–701999.",
    "- Validation: 40 fixed shifts, seeds 900000–900039. Checkpoints and planner batching are selected only here. Refinement was chosen after observing validation weakness, before opening the test set.",
    "- Refinement resets replay and Adam state, uses a 0.0001 learning rate, and retains an earlier checkpoint if validation gets worse.",
    f"- Test: {report['test_days']} fresh shifts per condition, starting at 1000000, 1001000, 1002000, and 1003000. Every policy faces matched exogenous streams.",
    "- Replay examples: 2000001, 2000002, 2000003. These were fixed independently of results.",
    "- Guided exploration: 70% of exploratory actions use earliest-deadline dispatch, 30% are random feasible commands. Evaluation has no heuristic fallback.",
    "- Neural checkpoints contain learned weights, not demonstration recordings. The selected demo model is chosen by validation reward.",
    "",
    "## Results",
    "",
    "Percentages below average per-shift ratios. Their denominator is all requested orders, including rejected and unfinished work. Priority service uses all priority requests. Reward is in synthetic points.",
]
for scenario, title in zip(report["summary"], titles):
    lines += [
        "",
        f"### {title}",
        "",
        "| Policy | Reward | On time | Completed | Priority on time | Distance / delivery |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, label in zip(keys, labels):
        r = report["summary"][scenario][k]
        lines.append(
            f"| {label} | {r['reward']:.2f} | {100 * r['on_time_rate']:.1f}% | {100 * r['completion_rate']:.1f}% | {100 * r['priority_service']:.1f}% | {r['distance_per_delivery']:.2f} |"
        )
    c = report["comparisons"][scenario]["planner"]
    lo, hi = c["paired_scenario_ci"]
    lines += [
        "",
        f"Neural minus planner reward: **{c['difference']:+.2f}**, paired scenario-bootstrap 95% interval **[{lo:+.2f}, {hi:+.2f}]**.",
    ]
lines += [
    "",
    "## Interpretation",
    "",
    "The route planner is stronger than the neural agent in the standard scenario in this run. I am retaining the result rather than presenting a more complex model as an automatic improvement. The experiment demonstrates a working RL environment and evaluation pipeline; it does not establish that this DQN should replace a planner.",
    "",
    "The original one-vehicle results belong to a different task. Its 87% on-time result cannot be carried over to the fleet version. More vehicles, uncertain travel, charging, service times and a different reward make this a harder problem.",
    "",
    "The next learning experiments would be a more structured state representation, a hierarchical policy that selects dispatch strategies, and a larger predeclared training budget. Those are future work, not completed features.",
    "",
    "## Uncertainty and scope",
    "",
    "Intervals use 2,000 paired resamples of scenario-level differences after averaging the three neural replicas. They condition on those trained policies and this simulator. They do not include full training or simulator uncertainty. The report also retains each replica separately. Comparisons are descriptive, without multiplicity adjustment.",
    "",
    "No real demand, road network, driver data, or business savings were used or measured. The route planner is not a globally optimal solver. The model uses known operating parameters and a fixed-size state encoding. See ENVIRONMENT.md for the information boundaries.",
    "",
    "## Verification",
    "",
    "Tests cover battery conservation, parcel accounting, simultaneous travel, service timing, action masks, model save/load, neural gradient updates, API validation, fresh reproducible simulations, and the original Gymnasium adapter. API requests are tested directly. Local HTML browser navigation is blocked in this execution environment, so browser end-to-end playback has not been verified. The static assets are rendered from simulator data and visually inspected.",
    "",
    "Raw episode rows, learning curves, selected checkpoints, training settings, and SHA-256 checksums are included under advanced/.",
]
(ROOT / "docs/v2/RESULTS.md").write_text("\n".join(lines) + "\n")
print("Fleet demo, assets and report built.")
