"""M0 — robustness probe: adjudicate the primitive set empirically (build plan M0).

Loads real harmonised grids, applies synthetic nuisance perturbations (gain, SO(3) rotation,
yaw-only rotation, anti-aliased resample, mild uniform time-warp), computes candidate feature
families before vs after, and reports, per family:

  * INVARIANCE      — mean cosine similarity + relative-L2 drift under each perturbation
  * DISCRIMINABILITY — subject-disjoint within-dataset kNN balanced accuracy, and
                       leave-one-dataset-out kNN balanced accuracy (the cross-config axis)

It also doubles as the A3-target separation check (EVIDENCE_ENGINE.md §5.2.3): boxplots of
cadence and per-band eigen-ratios across activities — the grounding targets must visibly
separate walk/run/sit/stairs or they are too noisy at this band resolution.

Gate: keep only families that are BOTH invariant to the nuisance AND still separate activities.

Run (needs scipy/matplotlib — use the legacy venv):
  /home/alex/code/HALO/legacy_code/.venv/bin/python -m training.tokenizer.probe_robustness
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sps

from data.scripts.eda.grid_io import GridRef, discover_grids, sample_indices, triad_indices

# ----------------------------------------------------------------------------------------------
# Probe configuration (all magic numbers live here)
# ----------------------------------------------------------------------------------------------
SEED = 20260717

# Diverse gravity-present streams: pocket / wrist / waist / pocket (kuhar & uci_har are
# gravity-removed and deliberately excluded — rotation physics needs gravity present).
STREAMS: tuple[tuple[str, str], ...] = (
    ("motionsense", "phone_front_pocket"),
    ("pamap2", "watch_wrist"),
    ("realworld", "phone_waist"),
    ("shoaib", "phone_right_pocket"),
)

# Locomotion + static labels; per stream we use the intersection with its own vocabulary.
PROBE_LABELS: tuple[str, ...] = (
    "walking", "jogging", "running", "sitting", "standing",
    "walking_upstairs", "walking_downstairs", "lying", "cycling",
)
WINDOWS_PER_LABEL = 40

# Physical-Hz analysis bands (shared across families). HI band only in the raw/grav energy
# families — under a half-rate resample it is partially truncated, which is honest signal.
BANDS: tuple[tuple[float, float], ...] = ((0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0))
HI_BAND: tuple[float, float] = (12.0, 25.0)
ACTIVE_BANDS = BANDS[1:]  # bands used for eigen-ratios / shape / coherence (DC excluded)

# Perturbation parameters
GAIN_RANGE = (0.7, 1.3)          # uniform per-window gain
YAW_RANGE_DEG = (30.0, 330.0)    # yaw angle about the estimated gravity axis
RESAMPLE_HZ = 30.0               # anti-aliased downsample target
TIMEWARP_ALPHAS = (0.9, 1.1)     # mild uniform speed factors (alpha>1 = faster playback)

GRAVITY_MIN_G = 0.5              # |lowpass acc| below this => gravity absent (skip align).
                                 # The corpus is unit-canonicalized to g (|gravity| ~= 1.0).
CADENCE_LAG_S = (0.25, 2.0)      # autocorr peak search window (0.5–4 Hz cadence)
CADENCE_MIN_STRENGTH = 0.30      # periodicity gate for cadence validity
CADENCE_MIN_MOTION_G = 0.03      # dynamic |acc| std floor (g) — static windows have no cadence
KNN_K = 5
EPS = 1e-12

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "training" / "tokenizer" / "outputs" / "m0_probe"


def _rng(*key: object) -> np.random.Generator:
    digest = hashlib.blake2b(":".join(map(str, key)).encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


# ----------------------------------------------------------------------------------------------
# Small DSP helpers
# ----------------------------------------------------------------------------------------------
def _band_power(x: np.ndarray, rate: float, lo: float, hi: float) -> np.ndarray:
    """Mean power of x (T, C) in [lo, hi) Hz, per channel. Rate-invariant (per-sample mean)."""
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / rate)
    sel = (freqs >= lo) & (freqs < hi)
    return (np.abs(spec[sel]) ** 2).sum(axis=0) / (x.shape[0] ** 2)


def _bandpass(x: np.ndarray, rate: float, lo: float, hi: float) -> np.ndarray:
    """FFT-mask bandpass of x (T, C)."""
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / rate)
    spec[(freqs < lo) | (freqs >= hi)] = 0.0
    return np.fft.irfft(spec, n=x.shape[0], axis=0)


def _gravity_vector(acc: np.ndarray, rate: float) -> np.ndarray | None:
    """Estimated gravity direction from the <0.5 Hz component; None if gravity absent."""
    g = _bandpass(acc, rate, 0.0, 0.5).mean(axis=0)
    return g if float(np.linalg.norm(g)) >= GRAVITY_MIN_G else None


def _rotation_to_z(u: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix taking unit vector u onto +z."""
    ez = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, ez)
    s, c = float(np.linalg.norm(v)), float(u @ ez)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s**2)


def _axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation by theta about a unit axis."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def _random_so3(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation matrix from a random unit quaternion."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _magnitude(x: np.ndarray) -> np.ndarray:
    m = np.linalg.norm(x, axis=1)
    return m - m.mean()


# ----------------------------------------------------------------------------------------------
# Perturbations: (acc, gyro, rate, rng) -> (acc', gyro', rate', cadence_factor)
# cadence_factor = analytic multiplier the true cadence undergoes (1.0 = invariant expected).
# ----------------------------------------------------------------------------------------------
def perturb_gain(acc, gyro, rate, rng):
    g = rng.uniform(*GAIN_RANGE)
    return acc * g, None if gyro is None else gyro * g, rate, 1.0


def perturb_rot_so3(acc, gyro, rate, rng):
    r = _random_so3(rng)
    return acc @ r.T, None if gyro is None else gyro @ r.T, rate, 1.0


def perturb_rot_yaw(acc, gyro, rate, rng):
    g = _gravity_vector(acc, rate)
    if g is None:  # no gravity axis to define yaw — leave unperturbed (excluded via zero drift)
        return acc, gyro, rate, 1.0
    theta = np.deg2rad(rng.uniform(*YAW_RANGE_DEG))
    r = _axis_rotation(g / np.linalg.norm(g), theta)
    return acc @ r.T, None if gyro is None else gyro @ r.T, rate, 1.0


def perturb_resample(acc, gyro, rate, rng):
    # Downsample to HALF the stream's NATIVE rate (polyphase, anti-aliased). Half is always a real
    # downsample across the corpus's 20/50/100 Hz native streams, whereas a fixed 30 Hz target would
    # UPSAMPLE the 20 Hz streams. The features are physical-Hz, so this is the rate-invariance stress.
    acc2 = sps.resample_poly(acc, 1, 2, axis=0)
    gyro2 = None if gyro is None else sps.resample_poly(gyro, 1, 2, axis=0)
    return acc2, gyro2, rate / 2.0, 1.0


def perturb_timewarp(acc, gyro, rate, rng):
    alpha = TIMEWARP_ALPHAS[int(rng.integers(len(TIMEWARP_ALPHAS)))]
    # y(t) = x(alpha * t): resample_poly(x, up, down) plays back at down/up speed.
    from fractions import Fraction
    frac = Fraction(alpha).limit_denominator(20)
    acc2 = sps.resample_poly(acc, frac.denominator, frac.numerator, axis=0)
    gyro2 = None if gyro is None else sps.resample_poly(gyro, frac.denominator, frac.numerator, axis=0)
    return acc2, gyro2, rate, alpha  # frequencies (cadence) scale by alpha


PERTURBATIONS = OrderedDict([
    ("gain", perturb_gain),
    ("rot_so3", perturb_rot_so3),
    ("rot_yaw", perturb_rot_yaw),
    ("resample_half", perturb_resample),
    ("time_warp", perturb_timewarp),
])


# ----------------------------------------------------------------------------------------------
# Feature families: (acc, gyro, rate) -> np.ndarray (fixed dim, NaN = invalid entry) or None
# ----------------------------------------------------------------------------------------------
def feat_raw_band_energy(acc, gyro, rate):
    """Baseline (expected fragile): log per-channel energy per physical band."""
    chans = acc if gyro is None else np.concatenate([acc, gyro], axis=1)
    out = [np.log10(_band_power(chans, rate, lo, hi) + EPS) for lo, hi in (*BANDS, HI_BAND)]
    return np.concatenate(out)


def feat_grav_band_energy(acc, gyro, rate):
    """Gravity-aligned, yaw-pooled: per band, [vertical, horizontal-pooled] x modality."""
    g = _gravity_vector(acc, rate)
    r = _rotation_to_z(g / np.linalg.norm(g)) if g is not None else np.eye(3)
    acc_a = acc @ r.T
    gyro_a = None if gyro is None else gyro @ r.T
    feats = []
    for lo, hi in (*BANDS, HI_BAND):
        for tri in (acc_a,) if gyro_a is None else (acc_a, gyro_a):
            p = _band_power(tri, rate, lo, hi)
            feats += [np.log10(p[2] + EPS), np.log10(p[0] + p[1] + EPS)]  # vert, horiz-pooled
    return np.asarray(feats)


def feat_eigen_ratios(acc, gyro, rate):
    """Per-band accel-triad covariance eigen-ratios: linearity / planarity / isotropy.
    Rotation-invariant (C -> R C R^T) and gain-invariant (ratios cancel g^2)."""
    feats = []
    for lo, hi in ACTIVE_BANDS:
        band = _bandpass(acc, rate, lo, hi)
        lam = np.sort(np.linalg.eigvalsh(np.cov(band.T)))[::-1]  # l1 >= l2 >= l3
        if lam[0] < 1e-10:
            feats += [np.nan, np.nan, np.nan]  # band ~empty: invalid, not zero
        else:
            feats += [(lam[0] - lam[1]) / lam[0], (lam[1] - lam[2]) / lam[0], lam[2] / lam[0]]
    return np.asarray(feats)


def feat_coherence(acc, gyro, rate):
    """Accel<->gyro magnitude coherence per band (rotation-invariant via magnitudes)."""
    if gyro is None:
        return None
    nper = min(int(2 * rate), acc.shape[0])
    f, cxy = sps.coherence(_magnitude(acc), _magnitude(gyro), fs=rate, nperseg=nper)
    out = []
    for lo, hi in ACTIVE_BANDS:
        sel = (f >= lo) & (f < hi)
        out.append(float(cxy[sel].mean()) if sel.any() else np.nan)
    return np.asarray(out)


def feat_spectral_shape(acc, gyro, rate):
    """Relative band-energy distribution + centroid + entropy of |acc| (gain+rot invariant)."""
    mag = _magnitude(acc)[:, None]
    powers = np.array([_band_power(mag, rate, lo, hi)[0] for lo, hi in ACTIVE_BANDS])
    total = powers.sum() + EPS
    rel = powers / total
    freqs_mid = np.array([(lo + hi) / 2 for lo, hi in ACTIVE_BANDS])
    centroid = float((rel * freqs_mid).sum())
    entropy = float(-(rel * np.log(rel + EPS)).sum())
    return np.concatenate([rel, [centroid, entropy]])


def feat_cadence(acc, gyro, rate):
    """[log2 cadence Hz, periodicity strength]; cadence NaN when aperiodic (validity mask).
    Validity needs BOTH periodicity strength AND real motion energy — near-static windows
    autocorrelate on drift and would otherwise fabricate a cadence."""
    mag = _magnitude(acc)
    if float(mag.std()) < CADENCE_MIN_MOTION_G:
        return np.array([np.nan, 0.0])
    ac = np.correlate(mag, mag, mode="full")[len(mag) - 1:]
    ac = ac / (ac[0] + EPS)
    lo, hi = int(CADENCE_LAG_S[0] * rate), min(int(CADENCE_LAG_S[1] * rate), len(ac) - 1)
    if hi <= lo:
        return np.array([np.nan, 0.0])
    lag = lo + int(np.argmax(ac[lo:hi]))
    strength = float(ac[lag])
    if strength < CADENCE_MIN_STRENGTH:
        return np.array([np.nan, strength])
    return np.array([np.log2(rate / lag), strength])


def feat_invariant_union(acc, gyro, rate):
    """The M1 candidate: concat of every invariant family (raw excluded on purpose)."""
    parts = [feat_grav_band_energy(acc, gyro, rate), feat_eigen_ratios(acc, gyro, rate),
             feat_spectral_shape(acc, gyro, rate), feat_cadence(acc, gyro, rate)]
    coh = feat_coherence(acc, gyro, rate)
    if coh is not None:
        parts.append(coh)
    return np.concatenate(parts)


FAMILIES = OrderedDict([
    ("raw_band_energy", feat_raw_band_energy),
    ("grav_band_energy", feat_grav_band_energy),
    ("eigen_ratios", feat_eigen_ratios),
    ("coherence", feat_coherence),
    ("spectral_shape", feat_spectral_shape),
    ("cadence", feat_cadence),
    ("invariant_union", feat_invariant_union),
])
CADENCE_DIM = 0  # index of the log-cadence entry inside the cadence family


# ----------------------------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------------------------
@dataclass
class ProbeWindow:
    dataset: str
    label: str
    subject: str
    rate: float        # the stream's NATIVE rate (grids are native; perturbations/features use it)
    acc: np.ndarray    # (T, 3) float64
    gyro: np.ndarray | None


def collect_windows() -> list[ProbeWindow]:
    refs = {(r.dataset, r.stream): r for r in discover_grids("native")}
    windows: list[ProbeWindow] = []
    for dataset, stream in STREAMS:
        ref = refs[(dataset, stream)]
        acc_idx, gyro_idx = triad_indices(ref, "acc"), triad_indices(ref, "gyro")
        assert acc_idx is not None, f"{ref.key}: accel triad required"
        data = ref.load_data()
        present = set(ref.labels)
        for label in PROBE_LABELS:
            if label not in present:
                continue
            count = min(WINDOWS_PER_LABEL, sum(1 for l in ref.labels if l == label))
            for idx in sample_indices(ref, label, count, SEED):
                win = np.asarray(data[int(idx)], dtype=np.float64)
                windows.append(ProbeWindow(
                    dataset=dataset, label=label, subject=ref.subjects[int(idx)],
                    rate=float(ref.rate_hz),
                    acc=win[:, list(acc_idx)],
                    gyro=None if gyro_idx is None else win[:, list(gyro_idx)],
                ))
    return windows


# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------
def _drift(before: np.ndarray, after: np.ndarray) -> tuple[float, float] | None:
    """(cosine similarity, relative L2) over dims valid in both; None if nothing valid."""
    ok = np.isfinite(before) & np.isfinite(after)
    if ok.sum() < 1:
        return None
    b, a = before[ok], after[ok]
    nb, na = np.linalg.norm(b), np.linalg.norm(a)
    if nb < EPS and na < EPS:
        return 1.0, 0.0
    cos = float(b @ a / (nb * na + EPS))
    rel = float(np.linalg.norm(a - b) / (nb + EPS))
    return cos, rel


def invariance_table(windows: list[ProbeWindow]) -> dict[str, dict[str, dict[str, float]]]:
    table: dict[str, dict[str, dict[str, float]]] = {f: {} for f in FAMILIES}
    for pname, pfn in PERTURBATIONS.items():
        acc_scores: dict[str, list[tuple[float, float]]] = {f: [] for f in FAMILIES}
        cadence_err: list[float] = []
        for w_i, w in enumerate(windows):
            rng = _rng(SEED, pname, w_i)
            acc2, gyro2, rate2, cad_factor = pfn(w.acc, w.gyro, w.rate, rng)
            for fname, ffn in FAMILIES.items():
                before, after = ffn(w.acc, w.gyro, w.rate), ffn(acc2, gyro2, rate2)
                if before is None or after is None:
                    continue
                if fname == "cadence":
                    # analytic correction: true cadence scales by cad_factor under time-warp
                    b, a = before[CADENCE_DIM], after[CADENCE_DIM]
                    if np.isfinite(b) and np.isfinite(a):
                        cadence_err.append(abs((a - np.log2(cad_factor)) - b))
                    continue
                d = _drift(before, after)
                if d is not None:
                    acc_scores[fname].append(d)
        for fname, scores in acc_scores.items():
            if fname == "cadence":
                continue
            if scores:
                arr = np.asarray(scores)
                table[fname][pname] = {"cosine": float(arr[:, 0].mean()),
                                       "rel_l2": float(arr[:, 1].mean())}
        if cadence_err:
            err = float(np.mean(cadence_err))
            # map |Δ log2 cadence| onto the same fields: rel_l2 = mean octave error
            table["cadence"][pname] = {"cosine": float(np.exp(-err)), "rel_l2": err}
    return table


def _feature_matrix(windows: list[ProbeWindow], fname: str) -> tuple[np.ndarray, np.ndarray]:
    ffn = FAMILIES[fname]
    rows, keep = [], []
    for i, w in enumerate(windows):
        v = ffn(w.acc, w.gyro, w.rate)
        if v is not None:
            rows.append(v)
            keep.append(i)
    x = np.asarray(rows)
    col_mean = np.nanmean(x, axis=0)
    x = np.where(np.isfinite(x), x, col_mean)         # NaN -> column mean
    mu, sd = x.mean(axis=0), x.std(axis=0) + EPS
    return (x - mu) / sd, np.asarray(keep)


def _knn_balanced_acc(train_x, train_y, test_x, test_y) -> float | None:
    labels = sorted(set(train_y) & set(test_y))
    if not labels:
        return None
    per_class = []
    for label in labels:
        sel = [i for i, y in enumerate(test_y) if y == label]
        hits = 0
        for i in sel:
            d = np.linalg.norm(train_x - test_x[i], axis=1)
            nn = np.argsort(d)[:KNN_K]
            votes = [train_y[j] for j in nn]
            hits += max(set(votes), key=votes.count) == label
        per_class.append(hits / len(sel))
    return float(np.mean(per_class))


def discriminability_table(windows: list[ProbeWindow]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for fname in FAMILIES:
        x, keep = _feature_matrix(windows, fname)
        ws = [windows[i] for i in keep]
        datasets = sorted({w.dataset for w in ws})
        # within-dataset, subject-disjoint (subjects split into two halves)
        within = []
        for ds in datasets:
            idx = [i for i, w in enumerate(ws) if w.dataset == ds]
            subjects = sorted({ws[i].subject for i in idx})
            _rng(SEED, "split", ds).shuffle(subjects)
            hold = set(subjects[: max(1, len(subjects) // 2)])
            tr = [i for i in idx if ws[i].subject not in hold]
            te = [i for i in idx if ws[i].subject in hold]
            if tr and te:
                score = _knn_balanced_acc(x[tr], [ws[i].label for i in tr],
                                          x[te], [ws[i].label for i in te])
                if score is not None:
                    within.append(score)
        # leave-one-dataset-out (the cross-config axis)
        xds = []
        for ds in datasets:
            tr = [i for i, w in enumerate(ws) if w.dataset != ds]
            te = [i for i, w in enumerate(ws) if w.dataset == ds]
            score = _knn_balanced_acc(x[tr], [ws[i].label for i in tr],
                                      x[te], [ws[i].label for i in te])
            if score is not None:
                xds.append(score)
        out[fname] = {"within_ba": float(np.mean(within)) if within else float("nan"),
                      "xdataset_ba": float(np.mean(xds)) if xds else float("nan")}
    return out


# ----------------------------------------------------------------------------------------------
# A3-target separation plots (cadence + eigen-ratios across activities)
# ----------------------------------------------------------------------------------------------
def a3_plots(windows: list[ProbeWindow], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l for l in PROBE_LABELS if sum(w.label == l for w in windows) >= 20]
    made: list[Path] = []

    def boxplot(values_by_label: dict[str, list[float]], title: str, ylabel: str, fname: str):
        data = [values_by_label[l] for l in labels if values_by_label.get(l)]
        names = [f"{l}\n(n={len(values_by_label[l])})" for l in labels if values_by_label.get(l)]
        if not data:
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.boxplot(data, tick_labels=names, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=120)
        plt.close(fig)
        made.append(path)

    # cadence (valid windows only)
    cad: dict[str, list[float]] = {l: [] for l in labels}
    for w in windows:
        if w.label in cad:
            v = feat_cadence(w.acc, w.gyro, w.rate)
            if np.isfinite(v[CADENCE_DIM]):
                cad[w.label].append(float(2 ** v[CADENCE_DIM]))
    boxplot(cad, "Cadence by activity (valid windows)", "cadence (Hz)", "a3_cadence.png")

    # eigen-ratios in the locomotion band (1.5–3 Hz)
    band_i = ACTIVE_BANDS.index((1.5, 3.0))
    for d, dim in enumerate(("linearity", "planarity", "isotropy")):
        vals: dict[str, list[float]] = {l: [] for l in labels}
        for w in windows:
            if w.label in vals:
                v = feat_eigen_ratios(w.acc, w.gyro, w.rate)[band_i * 3 + d]
                if np.isfinite(v):
                    vals[w.label].append(float(v))
        boxplot(vals, f"Eigen-ratio {dim} (1.5-3 Hz band) by activity", dim, f"a3_eigen_{dim}.png")
    return made


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def write_report(inv, disc, plots, n_windows, out_dir: Path) -> Path:
    lines = [
        "# M0 robustness probe — invariance vs discriminability",
        "",
        f"Probe sample: {n_windows} windows from {len(STREAMS)} streams "
        f"({', '.join(f'{d}/{s}' for d, s in STREAMS)}), seed {SEED}.",
        "",
        "## Invariance (mean cosine similarity / mean relative-L2 drift; cadence = exp(-|octave err|) / octave err)",
        "",
        "| family | " + " | ".join(PERTURBATIONS) + " |",
        "|---|" + "---|" * len(PERTURBATIONS),
    ]
    for fname in FAMILIES:
        cells = []
        for pname in PERTURBATIONS:
            cell = inv[fname].get(pname)
            cells.append(f"{cell['cosine']:.3f} / {cell['rel_l2']:.3f}" if cell else "n/a")
        lines.append(f"| {fname} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Discriminability (kNN balanced accuracy)",
        "",
        "| family | within-dataset (subject-disjoint) | cross-dataset (leave-one-out) |",
        "|---|---|---|",
    ]
    for fname in FAMILIES:
        d = disc[fname]
        lines.append(f"| {fname} | {d['within_ba']:.3f} | {d['xdataset_ba']:.3f} |")
    lines += [
        "",
        "Notes: `raw_band_energy` is the deliberate fragile baseline. `time_warp` is a",
        "deformation-stability check (smooth drift expected, exact invariance NOT expected,",
        "except cadence which is analytically corrected by the warp factor). The `12-25 Hz`",
        "band is truncated under `resample_half` by Nyquist — drift there is honest.",
        "",
        "## A3-target separation plots",
        "",
    ]
    lines += [f"![{p.stem}]({p.name})" for p in plots]
    lines.append("")
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("collecting probe windows ...", flush=True)
    windows = collect_windows()
    counts: dict[str, int] = {}
    for w in windows:
        counts[f"{w.dataset}:{w.label}"] = counts.get(f"{w.dataset}:{w.label}", 0) + 1
    print(f"  {len(windows)} windows: {counts}", flush=True)

    print("scoring invariance under perturbations ...", flush=True)
    inv = invariance_table(windows)
    print("scoring discriminability (kNN) ...", flush=True)
    disc = discriminability_table(windows)
    print("rendering A3 separation plots ...", flush=True)
    plots = a3_plots(windows, args.out)

    (args.out / "scores.json").write_text(json.dumps(
        {"invariance": inv, "discriminability": disc, "n_windows": len(windows),
         "streams": [f"{d}/{s}" for d, s in STREAMS], "seed": SEED}, indent=2))
    report = write_report(inv, disc, plots, len(windows), args.out)
    print(f"report: {report}", flush=True)


if __name__ == "__main__":
    main()
