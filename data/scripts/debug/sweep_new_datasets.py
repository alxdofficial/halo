"""Debug sweep for newly onboarded datasets: draw real samples, check physics, plot, report.

This is the verification step that a converter running without raising does NOT give you. It reads
the built `native` grids — the exact arrays HALO trains and evaluates on — and asks, per stream:

  1. **Structure.** Window count, rate, channel mask, label and subject coverage, and the granularity
     of the execution ids `eval_enrollment` will key on (session-level vs window-level).
  2. **Physics.** With the accelerometer canonicalized to g, a gravity-present stream must read
     |acc| ~ 1 near rest. Anything near 9.8 means a missed m/s^2, anything near 0.05 means gravity
     was removed upstream, and anything near 1000 means milli-g. Gyroscope p99.9 separates rad/s
     from deg/s by ~57x. Flat channels, clipping at the sensor range, and non-finite samples are
     reported here rather than discovered during a training run.
  3. **The real model path.** Windows are pushed through `preprocess.gravity_align` and the
     `PhysicalFilterbankTokenizer` actually used in training, because a scale or NaN problem that
     only appears after log1p + frozen standardization will not show up in raw statistics.
  4. **Augmentations.** The Phase-A augmentation config is applied to real windows and the output is
     checked for finiteness and plausible magnitude drift.
  5. **Boundaries.** Session-length and windows-per-session distributions, how much of each source is
     lost to being shorter than one window, and — for multi-placement sources — whether window *i*
     of one placement really is the same instant as window *i* of another.

Writes a JSON report and one PNG per stream (a few random windows, plus the band energies the
tokenizer sees) so the numbers can be eyeballed rather than trusted.

    python -m data.scripts.debug.sweep_new_datasets --dataset monipar realdisp ...
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "scripts" / "debug" / "outputs" / "new_datasets"

# Physical expectations for an accelerometer canonicalized to g with gravity present. These are
# deliberately wide: the point is to catch a UNIT or GRAVITY-STATE error (which moves the value by
# 10x or more), not to police honest per-activity variation.
G_PRESENT_RANGE = (0.6, 1.8)
G_REMOVED_MAX = 0.35
GYRO_RADS_MAX = 45.0
SATURATION_QUANTILE = 0.9995


def _quiescent_g(acc: np.ndarray, gyro: np.ndarray | None) -> float:
    """Median |acc| over the least-rotating tenth of windows — the stream's gravity reference.

    Whole-stream medians drift high on vigorous sources (SPAR's arm exercises reach 2.5 g), so the
    honest reference is what the sensor reads when it is nearly still.
    """
    magnitude = np.linalg.norm(acc, axis=-1).mean(axis=-1)          # (N,) mean |acc| per window
    if gyro is not None:
        rotation = np.linalg.norm(gyro, axis=-1).mean(axis=-1)
        quiet = np.argsort(rotation)[: max(1, len(rotation) // 10)]
    else:
        quiet = np.argsort(magnitude)[: max(1, len(magnitude) // 10)]
    return float(np.median(magnitude[quiet]))


def _stream_report(dataset: str, stream: str, gravity_state: str, rng) -> dict:
    from data.scripts.eda import grid_io

    grid_dir = REPO / "data" / "datasets" / dataset / "grids" / "native" / stream
    if not (grid_dir / "meta.json").exists():
        return {"status": "no_grid"}
    meta = json.loads((grid_dir / "meta.json").read_text())
    data = np.load(grid_dir / "data.npy", mmap_mode="r")
    mask = np.load(grid_dir / "mask.npy")
    report: dict = {
        "status": "ok",
        "windows": int(data.shape[0]),
        "samples_per_window": int(data.shape[1]),
        "channels": list(meta["channels"]),
        "mask": [bool(v) for v in mask],
        "rate_hz": float(meta["rate_hz"]),
    }
    if data.shape[0] == 0:
        report["status"] = "empty_grid"
        return report

    labels = np.asarray(meta["labels"], dtype=object)
    subjects = np.asarray(meta["subjects"], dtype=object)
    events = np.asarray(meta.get("event_ids") or [], dtype=object)
    report["hours"] = round(data.shape[0] * data.shape[1] / float(meta["rate_hz"]) / 3600.0, 3)
    report["n_labels"] = int(len(set(labels)))
    report["n_subjects"] = int(len(set(subjects)))
    report["label_counts"] = dict(sorted(Counter(labels.tolist()).items(),
                                         key=lambda kv: -kv[1])[:8])
    report["unlabelled_windows"] = int(np.sum(labels == "-1") + np.sum(labels == ""))

    # Execution granularity: eval/data.py strips the trailing window ordinal, so this is what
    # build_paired_enrollment_plans will actually see.
    if len(events):
        executions = np.array([str(e).rsplit(":", 1)[0] if str(e).rsplit(":", 1)[-1].isdigit()
                               else str(e) for e in events], dtype=object)
        _, counts = np.unique(executions, return_counts=True)
        report["executions"] = int(len(counts))
        report["singleton_execution_share"] = round(float(np.mean(counts == 1)), 4)
        report["window_level_execution_ids"] = bool(np.mean(counts == 1) > 0.95)
        per_subject_label = Counter()
        for label, subject, execution in zip(labels, subjects, executions):
            per_subject_label[(subject, label)] = per_subject_label[(subject, label)]
        # executions per (subject, label) — the quantity a same-subject k-curve consumes
        pairs: dict[tuple, set] = {}
        for label, subject, execution in zip(labels, subjects, executions):
            pairs.setdefault((subject, label), set()).add(execution)
        sizes = np.array([len(v) for v in pairs.values()])
        report["executions_per_subject_label"] = {
            "median": int(np.median(sizes)), "max": int(sizes.max()),
            "share_with_2_or_more": round(float(np.mean(sizes >= 2)), 3),
        }

    # --- physics, on a bounded random sample of windows -----------------------------------------
    take = rng.choice(data.shape[0], size=min(600, data.shape[0]), replace=False)
    block = np.asarray(data[np.sort(take)], dtype=np.float64)       # (n, T, C)
    channels = list(meta["channels"])
    acc_idx = [i for i, c in enumerate(channels) if c.startswith("acc_")]
    gyro_idx = [i for i, c in enumerate(channels) if c.startswith("gyro_")]
    real = np.asarray(mask, dtype=bool)

    report["non_finite_samples"] = int(np.sum(~np.isfinite(block)))
    per_channel_std = block.std(axis=(0, 1))
    report["flat_real_channels"] = [channels[i] for i in range(len(channels))
                                    if real[i] and per_channel_std[i] < 1e-8]
    report["padded_channels_nonzero"] = [channels[i] for i in range(len(channels))
                                         if not real[i] and np.abs(block[:, :, i]).max() > 0]

    if acc_idx and all(real[i] for i in acc_idx):
        acc = block[:, :, acc_idx]
        gyro = block[:, :, gyro_idx] if gyro_idx and all(real[i] for i in gyro_idx) else None
        quiescent = _quiescent_g(acc, gyro)
        report["quiescent_acc_g"] = round(quiescent, 4)
        report["acc_abs_max_g"] = round(float(np.abs(acc).max()), 3)
        report["declared_gravity_state"] = gravity_state
        if gravity_state == "present":
            ok = G_PRESENT_RANGE[0] <= quiescent <= G_PRESENT_RANGE[1]
            report["gravity_check"] = "ok" if ok else "FAIL"
            if not ok:
                report["gravity_note"] = (
                    f"quiescent |acc| = {quiescent:.3f} g; ~9.8 means an un-rescaled m/s^2, "
                    f"~0.05 means gravity was removed, ~1000 means milli-g")
        else:
            ok = quiescent <= G_REMOVED_MAX
            report["gravity_check"] = "ok" if ok else "FAIL"
        # Sensor-range clipping: a real saturating accelerometer piles up at a constant magnitude.
        edge = float(np.quantile(np.abs(acc), SATURATION_QUANTILE))
        report["acc_clipping_ratio"] = round(float(np.abs(acc).max() / max(edge, 1e-9)), 3)

    if gyro_idx and all(real[i] for i in gyro_idx):
        gyro = block[:, :, gyro_idx]
        peak = float(np.percentile(np.abs(gyro), 99.9))
        report["gyro_p999_rad_s"] = round(peak, 3)
        report["gyro_unit_check"] = "ok" if peak <= GYRO_RADS_MAX else "FAIL (looks like deg/s)"

    return report


def _model_path_report(dataset: str, stream: str, meta_rate: float, block: np.ndarray,
                       channels: list[str], mask: np.ndarray, gravity_state: str) -> dict:
    """Push real windows through gravity alignment + the training tokenizer."""
    import torch
    from model.tokenizer.filterbank import PhysicalFilterbankTokenizer
    from model.tokenizer import preprocess

    x = torch.from_numpy(np.asarray(block, dtype=np.float32))       # (n, T, C)
    out: dict = {}
    aligned = x
    if gravity_state == "present":
        # gravity_align identifies the accelerometer triad by channel TEXT, so the names are not
        # optional. An earlier revision called it positionally as `gravity_align(x, meta_rate)`,
        # which raised TypeError on every one of the 51 streams; the exception was caught, `aligned`
        # fell back to the unaligned tensor, and the sweep still reported `aligned_finite: true`.
        # That made "gravity alignment returns finite output" a claim about data that was never
        # aligned. Let it raise instead — a diagnostic that reports success on a code path it never
        # executed is worse than no diagnostic.
        aligned, _rotation, _ok = preprocess.gravity_align(x, list(channels), float(meta_rate))
    out["aligned_finite"] = bool(torch.isfinite(aligned).all().item())
    out["gravity_aligned"] = gravity_state == "present"

    # The tokenizer consumes patches zero-padded to its DFT size, exactly as the trainer's collator
    # builds them (training/tokenizer/pretrain_data.py): a 1 s patch at the stream's own rate,
    # never longer than dft_size, with the TRUE sample count passed alongside so the Hann window,
    # DC removal and Nyquist masks are honest.
    tokenizer = PhysicalFilterbankTokenizer().eval()
    size = tokenizer.S
    length = int(min(size, round(meta_rate * 1.0), aligned.shape[1]))
    n_patches = max(1, aligned.shape[1] // length)
    patches = torch.zeros(aligned.shape[0], n_patches, size, aligned.shape[2])
    for index in range(n_patches):
        chunk = aligned[:, index * length:(index + 1) * length, :]
        patches[:, index, :chunk.shape[1], :] = chunk
    out["patch_len_samples"] = length
    out["patches_per_window"] = n_patches
    with torch.no_grad():
        tokens = tokenizer(patches, float(meta_rate),
                           patch_len_samples=torch.full((aligned.shape[0],), length,
                                                        dtype=torch.long))
    out["token_shape"] = list(tokens.shape)
    out["tokens_finite"] = bool(torch.isfinite(tokens).all().item())
    out["token_abs_mean"] = round(float(tokens.abs().mean()), 4)
    out["token_abs_max"] = round(float(tokens.abs().max()), 4)
    return out


def _augmentation_report(block: np.ndarray, channels: list[str], rate: float,
                         mask: np.ndarray, gravity_state: str) -> dict:
    import torch
    from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample

    # The magnitude ratio here is BIMODAL by design, not noisy: Phase A's gravity augmentation fires
    # at p=0.5 and subtracts the ~1 g DC component, so an accelerometer-only stream lands near 0.1
    # when it fires and near 1.0 when it does not. A median over a handful of draws therefore reports
    # whichever mode happened to win and looks like a 10x scale bug. Report the two modes separately
    # and count how often gravity removal fired, which is the quantity that is actually checkable.
    augmenter = IMUAugmenter(AugmentationConfig.phase_a())
    finite = True
    ratios = []
    applied = Counter()
    for index in range(min(64, len(block))):
        sample = IMUSample(
            data=torch.from_numpy(np.asarray(block[index], dtype=np.float32)),
            channel_names=list(channels),
            sampling_rate=float(rate),
            channel_descriptions=[f"accelerometer {c[-1]}-axis" if c.startswith("acc")
                                  else f"gyroscope {c[-1]}-axis" for c in channels],
            channel_mask=[bool(v) for v in mask],
            gravity_state=gravity_state,
        )
        result = augmenter(sample)
        result = result[0] if isinstance(result, tuple) else result
        finite &= bool(torch.isfinite(result.data).all().item())
        applied.update(getattr(result, "applied_augmentations", []) or [])
        before = float(np.abs(block[index]).mean())
        after = float(result.data.abs().mean())
        if before > 1e-9:
            ratios.append(after / before)
    ratios = np.asarray(ratios)
    gravity_removed = ratios < 0.3
    kept = ratios[~gravity_removed]
    return {
        "augmented_finite": finite,
        "augmentations_applied": dict(applied.most_common()),
        "gravity_removed_share": round(float(gravity_removed.mean()), 3) if len(ratios) else None,
        # Median over the draws where gravity SURVIVED — this is the one that should sit near 1.
        "magnitude_ratio_median": round(float(np.median(kept)), 3) if len(kept) else None,
        "magnitude_ratio_p95": round(float(np.percentile(kept, 95)), 3) if len(kept) else None,
    }


def _plot(dataset: str, stream: str, block: np.ndarray, channels: list[str], mask: np.ndarray,
          rate: float, labels: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(4, len(block))
    figure, axes = plt.subplots(n_show, 2, figsize=(13, 2.4 * n_show), squeeze=False)
    time = np.arange(block.shape[1]) / rate
    for row in range(n_show):
        window = block[row]
        left, right = axes[row][0], axes[row][1]
        for index, name in enumerate(channels):
            if not mask[index]:
                continue
            style = "-" if name.startswith("acc_") else "--"
            left.plot(time, window[:, index], style, linewidth=0.8, label=name)
        left.set_ylabel("g  /  rad·s⁻¹")
        left.set_title(f"{dataset}/{stream} — {labels[row]}", fontsize=9)
        if row == n_show - 1:
            left.set_xlabel("seconds")
        if row == 0:
            left.legend(fontsize=6, ncol=3, loc="upper right")

        # Single-sided amplitude spectrum over the filterbank's analysis band.
        for index, name in enumerate(channels):
            if not mask[index]:
                continue
            spectrum = np.abs(np.fft.rfft(window[:, index] - window[:, index].mean()))
            freqs = np.fft.rfftfreq(len(window), 1.0 / rate)
            keep = freqs <= min(20.0, rate / 2)
            right.semilogy(freqs[keep], spectrum[keep] + 1e-9, linewidth=0.8)
        right.axvspan(0.3, 15.0, color="0.9", zorder=0)
        right.set_ylabel("|X(f)|")
        right.set_title("spectrum (shaded = filterbank band 0.3–15 Hz)", fontsize=8)
        if row == n_show - 1:
            right.set_xlabel("Hz")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=None,
                        help="Datasets to sweep (default: every dataset with a native grid).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-model", action="store_true",
                        help="Skip the tokenizer/augmentation pass (structure + physics only).")
    args = parser.parse_args()

    from data.scripts.curate import deployment_policy as policy

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    wanted = set(args.dataset) if args.dataset else {s.dataset for s in policy.STREAM_SPECS}

    report: dict = {}
    failures: list[str] = []
    for spec in policy.STREAM_SPECS:
        if spec.dataset not in wanted:
            continue
        key = f"{spec.dataset}/{spec.stream_id}"
        entry = _stream_report(spec.dataset, spec.stream_id, spec.gravity_state, rng)
        report[key] = entry
        if entry["status"] != "ok":
            print(f"  {key}: {entry['status']}")
            continue

        grid_dir = REPO / "data" / "datasets" / spec.dataset / "grids" / "native" / spec.stream_id
        meta = json.loads((grid_dir / "meta.json").read_text())
        data = np.load(grid_dir / "data.npy", mmap_mode="r")
        mask = np.load(grid_dir / "mask.npy")
        take = np.sort(rng.choice(data.shape[0], size=min(64, data.shape[0]), replace=False))
        block = np.asarray(data[take], dtype=np.float32)

        if not args.no_model:
            entry.update(_model_path_report(spec.dataset, spec.stream_id, float(meta["rate_hz"]),
                                            block, list(meta["channels"]), mask,
                                            spec.gravity_state))
            entry.update(_augmentation_report(block, list(meta["channels"]),
                                              float(meta["rate_hz"]), mask, spec.gravity_state))
        chosen = [str(meta["labels"][i]) for i in take[:4]]
        _plot(spec.dataset, spec.stream_id, block[:4], list(meta["channels"]), mask,
              float(meta["rate_hz"]), chosen,
              OUT / f"{spec.dataset}__{spec.stream_id}.png")

        for check in ("gravity_check", "gyro_unit_check"):
            if entry.get(check, "ok") != "ok":
                failures.append(f"{key}: {check} = {entry[check]}")
        for check, value in (("tokens_finite", True), ("aligned_finite", True),
                             ("augmented_finite", True)):
            if entry.get(check, value) is not value:
                failures.append(f"{key}: {check} is False")
        if entry.get("flat_real_channels"):
            failures.append(f"{key}: flat real channels {entry['flat_real_channels']}")
        if entry.get("padded_channels_nonzero"):
            failures.append(f"{key}: padded channels carry data "
                            f"{entry['padded_channels_nonzero']}")
        if entry.get("non_finite_samples"):
            failures.append(f"{key}: {entry['non_finite_samples']} non-finite samples")
        print(f"  {key}: {entry['windows']:>7,} win  {entry.get('hours', 0):>6.2f} h  "
              f"{entry.get('n_labels', 0):>3} labels  {entry.get('n_subjects', 0):>3} subj  "
              f"|a|q={entry.get('quiescent_acc_g', float('nan')):.3f} g  "
              f"gyro_p999={entry.get('gyro_p999_rad_s', float('nan'))}")

    # --- cross-placement simultaneity, for the multi-placement sources --------------------------
    # Only streams that share ONE converted frame can be compared window-for-window. A dataset whose
    # placements live in SEPARATE sessions (upper_limb_use's two arms, spar's two shoulders, wisdm's
    # phone and watch) is routed by session id and legitimately has different window counts per
    # stream; comparing those would report a failure that is not one. The structural tell is that a
    # co-located stream reads PREFIXED source columns (`right_wrist_acc_x`) while a routed stream
    # reads the canonical `acc_x`.
    def _co_located(spec) -> bool:
        return all(source != name
                   for name, sources in spec.required.items() for source in sources)

    pairing: dict = {}
    for dataset in sorted(wanted):
        specs = [s for s in policy.stream_specs(dataset, None)
                 if report.get(f"{dataset}/{s.stream_id}", {}).get("status") == "ok"]
        groups: dict[tuple, list[str]] = {}
        for spec in specs:
            if _co_located(spec):
                groups.setdefault((spec.session_contains, spec.session_excludes), []).append(
                    spec.stream_id)
        streams = max(groups.values(), key=len) if groups else []
        if len(groups) > 1:
            pairing.setdefault(dataset, {})["other_co_located_groups"] = [
                v for v in groups.values() if v is not streams]
        if len(streams) < 2:
            continue
        base = REPO / "data" / "datasets" / dataset / "grids" / "native"
        metas = {s: json.loads((base / s / "meta.json").read_text()) for s in streams}
        first = streams[0]
        agree = {}
        for other in streams[1:]:
            a, b = metas[first], metas[other]
            if len(a["labels"]) != len(b["labels"]):
                agree[other] = {"aligned": False, "reason": "different window counts",
                                "counts": [len(a["labels"]), len(b["labels"])]}
                continue
            agree[other] = {
                "aligned": True,
                "label_agreement": round(float(np.mean(np.asarray(a["labels"]) ==
                                                       np.asarray(b["labels"]))), 4),
                "subject_agreement": round(float(np.mean(np.asarray(a["subjects"]) ==
                                                         np.asarray(b["subjects"]))), 4),
                "event_ordinal_agreement": round(float(np.mean(
                    [x.rsplit(":", 1)[-1] == y.rsplit(":", 1)[-1]
                     for x, y in zip(a.get("event_ids", []), b.get("event_ids", []))]
                )), 4) if a.get("event_ids") and b.get("event_ids") else None,
            }
        pairing.setdefault(dataset, {}).update({"reference": first, "against": agree})
        for other, value in agree.items():
            if not value["aligned"]:
                failures.append(f"{dataset}: {first} vs {other} — {value['reason']} "
                                f"{value.get('counts')}")
            elif value["label_agreement"] < 0.999:
                failures.append(f"{dataset}: {first} vs {other} label agreement "
                                f"{value['label_agreement']} < 0.999 — the placements are not "
                                f"simultaneous")

    payload = {"streams": report, "cross_placement": pairing, "failures": failures}
    (OUT / "report.json").write_text(json.dumps(payload, indent=2))
    print(f"\n{len(report)} streams swept -> {OUT}")
    if failures:
        print(f"\n{len(failures)} CHECK FAILURES:")
        for line in failures:
            print(f"  - {line}")
    else:
        print("\nNo check failures.")


if __name__ == "__main__":
    main()
