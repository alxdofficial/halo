"""M2 augmentation-range visual inspection (A1 build discipline, §5.2.1).

Draw REAL windows through the actual training augmentation stack — one axis at a
time, forced on — and plot before/after so the chosen ranges can be eyeballed
before any training commits to them. Also renders the multi-scale patch-duration
grid (how many tokens each patch_seconds draw yields).

Run:  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.aug_inspect
"""

from __future__ import annotations

import random as stdlib_random
from pathlib import Path

import numpy as np
import torch

from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample
from data.scripts.eda.grid_io import discover_grids, sample_indices

SEED = 20260718
RATE = 60.0
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
DESCRIPTIONS = [
    "accelerometer x-axis at the front trouser pocket",
    "accelerometer y-axis at the front trouser pocket",
    "accelerometer z-axis at the front trouser pocket",
    "gyroscope x-axis at the front trouser pocket",
    "gyroscope y-axis at the front trouser pocket",
    "gyroscope z-axis at the front trouser pocket",
]
# the axes we inspect, each forced on alone (p=1) with the training-default ranges
AXES = ("rotation_3d", "rate", "time_warp", "channel_dropout", "scale", "jitter")
PATCH_SECONDS_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)
OUT = Path(__file__).resolve().parent / "outputs" / "m2_aug_inspect"


def one_axis_config(axis: str) -> AugmentationConfig:
    cfg = AugmentationConfig.none()
    spec = getattr(cfg, axis)
    spec.enabled = True
    spec.p = 1.0
    return cfg


def fetch_windows() -> list[tuple[str, torch.Tensor, float]]:
    refs = {r.dataset: r for r in discover_grids("native")}
    out = []
    for dataset, label in (("motionsense", "walking"), ("motionsense", "sitting"),
                           ("pamap2", "walking")):
        ref = refs[dataset]
        idx = int(sample_indices(ref, label, 1, SEED)[0])
        data = torch.tensor(np.asarray(ref.load_data()[idx]), dtype=torch.float32)
        out.append((f"{dataset}:{label}", data, float(ref.rate_hz)))   # native rate per window
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    windows = fetch_windows()
    made = []

    for axis in AXES:
        augmenter = IMUAugmenter(one_axis_config(axis))
        fig, axes_grid = plt.subplots(len(windows), 2, figsize=(12, 3 * len(windows)),
                                      sharex="col")
        for row, (name, data, rate) in enumerate(windows):
            stdlib_random.seed(SEED + row)
            np.random.seed(SEED + row)
            torch.manual_seed(SEED + row)
            sample = IMUSample(data=data.clone(), channel_names=list(CHANNELS),
                               sampling_rate=rate,
                               channel_descriptions=list(DESCRIPTIONS), label=name)
            aug = augmenter(sample)
            t0 = np.arange(data.shape[0]) / rate
            t1 = np.arange(aug.data.shape[0]) / aug.sampling_rate
            for col, (sig, t, title) in enumerate(
                ((data, t0, f"{name} — original ({rate:g} Hz)"),
                 (aug.data, t1, f"{axis} -> {aug.sampling_rate:g} Hz, "
                                f"{aug.data.shape[0]} samples"))
            ):
                ax = axes_grid[row][col]
                for c, style in ((0, "-"), (1, "-"), (2, "-")):
                    ax.plot(t, sig[:, c].numpy(), style, lw=0.7, alpha=0.8,
                            label=CHANNELS[c] if row == 0 else None)
                ax.set_title(title, fontsize=9)
                ax.set_ylabel("acc (g)", fontsize=8)
        axes_grid[0][0].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"augmentation axis: {axis} (forced on, training-default ranges)")
        fig.tight_layout()
        path = OUT / f"aug_{axis}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        made.append(path)

    # multi-scale patch grid: token count per patch_seconds draw
    name, data, rate = windows[0]
    fig, axs = plt.subplots(len(PATCH_SECONDS_GRID), 1, figsize=(12, 8), sharex=True)
    t = np.arange(data.shape[0]) / rate
    for ax, ps in zip(axs, PATCH_SECONDS_GRID):
        ax.plot(t, data[:, 2].numpy(), lw=0.7)
        n_patches = int(data.shape[0] / (ps * rate))
        for k in range(1, n_patches + 1):
            ax.axvline(k * ps, color="red", lw=0.8, alpha=0.6)
        ax.set_title(f"patch_seconds={ps} -> T={n_patches} time tokens "
                     f"({int(ps * rate)} samples/patch)", fontsize=9)
    axs[-1].set_xlabel("seconds")
    fig.suptitle(f"multi-scale patch-duration axis on {name} (acc_z)")
    fig.tight_layout()
    path = OUT / "patch_duration_grid.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    made.append(path)

    for p in made:
        print(p)


if __name__ == "__main__":
    main()
