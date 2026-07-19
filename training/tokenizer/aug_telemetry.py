"""Augmentation magnitude telemetry — how hard does each aug hit the signal?

For each augmentation (forced on alone, at its default_v2 params), measure its NUMERIC effect vs the
raw window over a sample of real native windows, so we can see whether any aug is over-powering the
data. Reports, per aug (medians unless noted):

  rel_L2   ||aug - raw||_F / ||raw||_F  on the overlapping region (shape-preserving augs; a rough
           "fraction of the signal changed"). ~0 = subtle, ~1 = the signal is mostly replaced.
  rms_x    RMS(aug) / RMS(raw) — amplitude scaling introduced.
  |dc|dHz  |cadence(aug) - cadence(raw)| in Hz — physics drift (should be ~0 for jitter/scale/
           rotation/rate; NON-zero for gravity-removal, which strips the low-frequency DC by design).
  struct   structural change (rate factor / fraction of samples kept / channels dropped).

Run: CUDA_VISIBLE_DEVICES='' /home/alex/code/HALO/legacy_code/.venv/bin/python \
       -m training.tokenizer.aug_telemetry [--n 256]
"""

from __future__ import annotations

import argparse
import random as stdlib_random

import numpy as np
import torch

from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample
from model.tokenizer.primitives import cadence
from training.tokenizer.pretrain_data import CHANNELS, GRAVITY_AUG_P, CorpusIndex

SEED = 20260718
# The physics/text augs to profile, each forced on ALONE at its default_v2 params.
PROFILE = ("window_crop", "rate", "rotation_3d", "gravity", "channel_dropout",
           "scale", "jitter", "channel_text_phrase", "channel_text_dropout")


def _one_aug_cfg(name: str) -> AugmentationConfig:
    cfg = AugmentationConfig.none()
    spec = getattr(cfg, name)
    spec.enabled = True
    spec.p = 1.0                                  # force it every time so we measure its full effect
    if name == "gravity":
        # keep the same removal strength the loader uses (only the fire-probability differs)
        spec.cutoff_hz = getattr(cfg, name).cutoff_hz
    return cfg


def _cad_hz(x: torch.Tensor, rate: float) -> float:
    c = cadence(x[:, :3].unsqueeze(0), rate)
    if not bool(c.valid[0]):
        return float("nan")
    return float(2.0 ** c.values[0, 0])           # log2-Hz -> Hz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256, help="windows sampled for the estimate")
    args = ap.parse_args()

    idx = CorpusIndex(max_per_stream=None, seed=SEED)
    rng = np.random.default_rng(SEED)
    keys = [idx.train[i] for i in rng.choice(len(idx.train), size=min(args.n, len(idx.train)),
                                             replace=False)]
    print(f"corpus: {idx.summary()} | profiling {len(keys)} windows\n")

    def raw_sample(k):
        ref = idx.refs[k.stream_i]
        w = torch.tensor(np.asarray(idx.refs[k.stream_i].load_data()[k.window_i], dtype=np.float32))
        return w, float(ref.rate_hz), [bool(m) for m in ref.mask]

    rows = []
    for name in PROFILE:
        aug = IMUAugmenter(_one_aug_cfg(name))
        rel_l2, rms_x, dcad, struct = [], [], [], []
        for k in keys:
            w, rate, cmask = raw_sample(k)
            stdlib_random.seed(hash((name, k.window_i)) & 0xFFFF)
            np.random.seed(hash((name, k.window_i)) & 0xFFFF)
            s = IMUSample(data=w.clone(), channel_names=list(CHANNELS), sampling_rate=rate,
                          channel_descriptions=["c"] * 6, label="a", dataset_name=ref_ds(idx, k),
                          channel_mask=cmask)
            a = aug(s)
            # structural
            if abs(a.sampling_rate - rate) > 1e-6:
                struct.append(("rate", a.sampling_rate / rate))
            elif a.data.shape[0] != w.shape[0]:
                struct.append(("kept", a.data.shape[0] / w.shape[0]))
            elif a.data.shape[1] != w.shape[1]:
                struct.append(("chdrop", 6 - a.data.shape[1]))
            # text-only augs change nothing numeric
            if name.startswith("channel_text"):
                continue
            # cadence drift (physics)
            c0, c1 = _cad_hz(w, rate), _cad_hz(a.data, a.sampling_rate)
            if np.isfinite(c0) and np.isfinite(c1):
                dcad.append(abs(c1 - c0))
            # signal-space rel-L2 + rms ratio, only where shapes line up (shape-preserving augs)
            if a.data.shape == w.shape:
                num = float((a.data - w).norm())
                den = float(w.norm()) + 1e-8
                rel_l2.append(num / den)
                rms_x.append(float(a.data.pow(2).mean().sqrt() / (w.pow(2).mean().sqrt() + 1e-8)))
        rows.append((name, rel_l2, rms_x, dcad, struct))

    def med(v):
        return float(np.median(v)) if v else float("nan")
    print(f"{'aug':22s} {'rel_L2':>8} {'rms_x':>7} {'|dcad|Hz':>9}  structural")
    print("-" * 70)
    for name, rel_l2, rms_x, dcad, struct in rows:
        if struct:
            kinds = {}
            for k, v in struct:
                kinds.setdefault(k, []).append(v)
            st = ", ".join(f"{k} med={med(vs):.2f}" for k, vs in kinds.items())
        elif name.startswith("channel_text"):
            st = "text-only (no numeric change)"
        else:
            st = "shape-preserving"
        print(f"{name:22s} {med(rel_l2):>8.3f} {med(rms_x):>7.3f} {med(dcad):>9.3f}  {st}")


def ref_ds(idx, k):
    return idx.refs[k.stream_i].dataset


if __name__ == "__main__":
    main()
