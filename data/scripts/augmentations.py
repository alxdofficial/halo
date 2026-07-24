"""
Augmentation strategies for IMU time series data.

Implements physically plausible augmentations for IMU sensor data following
research best practices from TS-TCC, PPDA, and recent literature (2023-2024).

Augmentations are divided into:
- Weak: jitter, scale, time_shift (preserve semantic meaning)
- Strong: time_warp, magnitude_warp, resample (more aggressive)
"""

import re
import torch
import numpy as np
from scipy import interpolate
from typing import Tuple, Optional, List

# Sensor-type token detector for triad location inference. Longest alternative first so
# 'accelerometer'/'accel' win over 'acc' (else 'acc' truncates 'accel' and mis-locates); the
# trailing \d* absorbs dual-range suffixes like pamap2's acc16/acc6.
_SENSOR_TOKEN_RE = re.compile(
    r'(accelerometer|accel|acc|gyroscope|gyro|magnetometer|magnet|mag|orientation|orient|ori)\d*')



# =============================================================================
# Unified, configurable augmentation system (V2)
# =============================================================================
# Every augmentation is switched on/off and tuned from a single
# AugmentationConfig, so it is obvious at a glance which augmentations are
# active. Physics/metadata-changing augmentations (gravity, rate, channel
# dropout) also update the per-sample channel description / sampling rate so the
# model's channel-text conditioning stays consistent with the augmented signal
# (the loader appends the "sampled at NHz" suffix from sample.sampling_rate).

import random as _random
from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from fractions import Fraction
from scipy import signal as _sps


# ---- Per-augmentation config specs (each has `enabled` + `p` + its params) ----
@dataclass
class JitterCfg:
    """Additive Gaussian sensor noise, scaled by each channel's local signal std."""
    enabled: bool = True
    p: float = 0.5
    sigma: float = 0.05


@dataclass
class ScaleCfg:
    """Per-channel amplitude scaling (gain/calibration variance)."""
    enabled: bool = True
    p: float = 0.5
    low: float = 0.9
    high: float = 1.1


@dataclass
class TimeShiftCfg:
    """Whole-window temporal (phase) shift."""
    enabled: bool = False
    p: float = 0.5
    max_ratio: float = 0.05


@dataclass
class TimeWarpCfg:
    """Non-linear time warp (cadence variation)."""
    enabled: bool = False
    p: float = 0.3
    n_knots: int = 4
    strength: float = 0.2


@dataclass
class MagnitudeWarpCfg:
    """Smooth per-channel amplitude modulation over time."""
    enabled: bool = False
    p: float = 0.3
    n_knots: int = 4
    strength: float = 0.3


@dataclass
class GravityCfg:
    """P1 — add/remove gravity. Subtracts a low-pass gravity estimate to
    manufacture the iOS `userAcceleration` (gravity-removed) representation the
    training corpus otherwise lacks; annotates the acc channel text accordingly."""
    enabled: bool = False
    p: float = 0.5
    cutoff_hz: float = 0.4   # gravity is quasi-DC; human motion energy is > ~0.5 Hz
    order: int = 2


@dataclass
class Rotation3dCfg:
    """P2b — full uniform-random SO(3) rotation of every co-located sensor triad
    (acc+gyro+mag share one rotation per body location). This is the placement/
    orientation-invariance lever (cf. UniMTS): the gravity DC vector rotates WITH the
    accel signal, so the model learns 'gravity can point any direction' rather than
    memorizing each dataset's fixed orientation. Principled ONLY because the filterbank
    now carries a signed DC feature (else full SO(3) scrambles an unrepresented gravity
    cue — the reason plain rotation was originally disabled). Gated on a gravity-present
    acc triad so normalized / gravity-removed data (recgym, iOS userAcceleration) is not
    rotated destructively."""
    enabled: bool = False
    p: float = 0.5
    require_gravity: bool = True   # skip locations whose acc triad is gravity-removed/normalized


@dataclass
class RateCfg:
    """P3 — anti-aliased resample to a random sampling rate (teaches rate-invariance).
    Updates sample.sampling_rate; the rate is conveyed NUMERICALLY to the filterbank
    (via the collate's per-sample ``rate``), NOT through text — the channel text carries no Hz
    token (rate/duration were stripped from the template), so descriptions stay byte-identical."""
    enabled: bool = False
    p: float = 0.5
    min_hz: float = 15.0
    max_hz: float = 100.0
    min_samples: int = 32   # skip if the resampled window would be shorter than this


@dataclass
class ChannelDropoutCfg:
    """P4 — drop a whole sensor group (e.g. gyro) so the model is robust to
    deployments that expose only an accelerometer. Updates channel list + text.

    Note: reduces a sample's channel count, so within a ChannelBucketBatchSampler
    bucket batches become channel-heterogeneous. This is correct (collate pads to
    the batch max and the per-sample channel_mask isolates padding) but slightly
    reduces the sampler's padding-efficiency; keep `p` modest for that reason."""
    enabled: bool = False
    p: float = 0.3
    groups: tuple = ("gyro",)   # channel-name substrings eligible for dropping


@dataclass
class WindowCropCfg:
    """P5 — random temporal crop: keep a contiguous sub-window of a random duration, so the model
    sees variable OBSERVATION LENGTHS (session-length invariance). The encoder is a set over patches
    with physical-time positions + a patch-padding mask, so a shorter window is simply FEWER real
    patches — nothing structural forces the 6 s corpus window. Layout-breaking (changes the token
    count), so it is HALO-only like rate/channel_dropout: a fixed-window baseline cannot ingest it."""
    enabled: bool = False
    p: float = 0.5
    min_frac: float = 0.5    # keep at least this fraction of the window's timesteps
    min_samples: int = 32    # never crop below this many samples (one resolvable patch)


@dataclass
class LabelTextCfg:
    """Label paraphrase: dataset-specific synonym swap + template wrapping (augment_label).
    Effective augmentation rate == `p` (the augmenter's outer gate; augment_label is called
    with rate=1.0 once selected)."""
    enabled: bool = False
    p: float = 0.8
    use_synonyms: bool = True
    use_templates: bool = True


@dataclass
class ChannelTextPhraseCfg:
    """Paraphrase each channel description: swap ONLY sensor-family / axis surface forms and
    wrap in a template. Placement, units, and gravity state are left verbatim, so the
    load-bearing semantics are preserved."""
    enabled: bool = False
    p: float = 0.5   # fraction of samples whose channel descriptions get paraphrased


@dataclass
class ChannelTextDropoutCfg:
    """Neutralize a random subset of channel descriptions (KEEP the signal) so the model is
    robust to unknown/missing placement metadata. Never neutralizes more than `max_frac`."""
    enabled: bool = False
    p: float = 0.15          # fraction of samples that get any channel-text neutralized
    max_frac: float = 0.5    # never neutralize more than this fraction of a sample's channels
    neutral: str = "an inertial sensor channel"


# Conservative, meaning-preserving substitutions for channel-description paraphrase. Only
# sensor-family + axis SURFACE FORMS are swapped; placement/units/gravity are never touched.
_CH_SYNONYMS = [
    (r"\baccelerometer\b", ["accelerometer", "accelerometer sensor"]),
    (r"\bacceleration\b", ["acceleration", "acceleration signal"]),
    (r"\bgyroscope\b", ["gyroscope", "gyro", "angular rate sensor"]),
    (r"\bangular velocity\b", ["angular velocity", "angular rate", "rotational velocity"]),
    (r"\bmagnetometer\b", ["magnetometer", "magnetic field sensor"]),
    (r"\bmagnetic field\b", ["magnetic field", "magnetic flux"]),
    (r"\bx-axis\b", ["x-axis", "x axis"]),
    (r"\by-axis\b", ["y-axis", "y axis"]),
    (r"\bz-axis\b", ["z-axis", "z axis"]),
    (r"\bmounted\b", ["mounted", "worn", "placed"]),
]
_CH_TEMPLATES = ["{}", "channel: {}", "sensor channel — {}", "signal from {}", "this channel measures {}"]


def _paraphrase_channel(desc: str) -> str:
    """Surface-form paraphrase of one channel description (sensor/axis synonyms + template).
    re.escape not needed — replacements are plain words; placement/units are never matched."""
    import re
    out = desc
    for pat, options in _CH_SYNONYMS:
        if re.search(pat, out, flags=re.I):
            out = re.sub(pat, _random.choice(options), out, flags=re.I)
    return _random.choice(_CH_TEMPLATES).format(out)


@dataclass
class AugmentationConfig:
    """Single source of truth for which augmentations run and how strong they are.

    Defaults reproduce the legacy behaviour (jitter + scale only). Presets:
      - AugmentationConfig()            -> legacy (jitter + scale)
      - AugmentationConfig.default_v2() -> P1-P4 curriculum ON (+ jitter + scale)
      - AugmentationConfig.none()       -> everything off
    Print `cfg.summary()` to see the ON/OFF table.
    """
    jitter: JitterCfg = field(default_factory=JitterCfg)
    scale: ScaleCfg = field(default_factory=ScaleCfg)
    time_shift: TimeShiftCfg = field(default_factory=TimeShiftCfg)
    time_warp: TimeWarpCfg = field(default_factory=TimeWarpCfg)
    magnitude_warp: MagnitudeWarpCfg = field(default_factory=MagnitudeWarpCfg)
    gravity: GravityCfg = field(default_factory=GravityCfg)
    rotation_3d: Rotation3dCfg = field(default_factory=Rotation3dCfg)
    rate: RateCfg = field(default_factory=RateCfg)
    channel_dropout: ChannelDropoutCfg = field(default_factory=ChannelDropoutCfg)
    window_crop: WindowCropCfg = field(default_factory=WindowCropCfg)
    # Text augmentations (unified here so ALL augmentation lives in one config).
    channel_text_phrase: ChannelTextPhraseCfg = field(default_factory=ChannelTextPhraseCfg)
    channel_text_dropout: ChannelTextDropoutCfg = field(default_factory=ChannelTextDropoutCfg)
    label_text: LabelTextCfg = field(default_factory=LabelTextCfg)

    # Application order: metadata/physics-changing first, then value-space, then TEXT last
    # (so channel-text augs see the final, physics-mutated channel set/descriptions).
    # rotation_3d runs BEFORE gravity (it needs gravity present in the acc to gate on);
    # rate runs after gravity/rotation.
    ORDER = ("window_crop", "channel_dropout", "rotation_3d", "gravity", "rate",
             "time_warp", "time_shift", "magnitude_warp", "scale", "jitter",
             "channel_text_phrase", "channel_text_dropout", "label_text")

    @classmethod
    def default_v2(cls) -> "AugmentationConfig":
        cfg = cls()
        cfg.gravity.enabled = True
        # Full SO(3) rotation is the placement-invariance lever; principled now that
        # the filterbank carries a signed DC/gravity feature.
        cfg.rotation_3d.enabled = True
        cfg.rate.enabled = True
        cfg.channel_dropout.enabled = True
        cfg.window_crop.enabled = True    # variable observation length (session-length invariance)
        cfg.channel_text_phrase.enabled = True
        cfg.channel_text_dropout.enabled = True
        cfg.label_text.enabled = True
        return cfg

    @classmethod
    def legacy(cls) -> "AugmentationConfig":
        """Only jitter + scale (pre-V2 effective behaviour)."""
        return cls()

    @classmethod
    def none(cls) -> "AugmentationConfig":
        cfg = cls()
        for name in cls.ORDER:
            getattr(cfg, name).enabled = False
        return cfg

    def summary(self) -> str:
        lines = ["Augmentation config (ON/OFF + params):"]
        for name in self.ORDER:
            spec = getattr(self, name)
            params = ", ".join(
                f"{f.name}={getattr(spec, f.name)}"
                for f in _dc_fields(spec) if f.name not in ("enabled", "p")
            )
            flag = "ON " if spec.enabled else "off"
            lines.append(f"  [{flag}] {name:16s} p={spec.p:<4} {params}")
        return "\n".join(lines)


@dataclass
class IMUSample:
    """Per-sample carrier threaded through the augmenter. Physics augmentations mutate
    sampling_rate / channel metadata and the TEXT augmentations mutate both legacy per-channel
    descriptions and, when supplied, factored role/sensor descriptions."""
    data: "torch.Tensor"              # (T, C)
    channel_names: List[str]
    sampling_rate: float
    channel_descriptions: List[str]   # base per-channel text (no Hz/window suffix)
    label: str = ""                   # raw activity label (input to label-text augmentation)
    dataset_name: str = ""            # for dataset-specific label synonyms
    label_text: str = ""              # augmented label text (output; defaults to raw label)
    channel_mask: Optional[List[bool]] = None   # True = REAL channel (False = zero-padded absent);
                                                # lets text dropout skip padded channels (F10b)
    role_descriptions: Optional[List[str]] = None
    sensor_descriptions: Optional[List[str]] = None
    sensor_id: Optional[List[int]] = None
    gravity_state: Optional[str] = None

    def __post_init__(self):
        if not self.label_text:
            self.label_text = self.label


def _gravity_present(triad: "np.ndarray", descs=None) -> bool:
    """True if an accelerometer triad still contains the gravity DC component.

    Prefers the DOCUMENTED gravity state (channel-description text is authoritative);
    only falls back to a hardened signal heuristic when the text is silent. The old
    pure DC/RMS ratio misfired on normalized data (e.g. recgym acc centered ~0.5, where
    dc/rms~1) and on low-motion gravity-removed data (e.g. kuhar static postures), so it
    now also requires the DC vector to be axis-concentrated (real gravity points ~down =
    one dominant axis; a uniform per-axis offset spreads across axes and is NOT gravity).
    """
    if descs:
        j = " ".join(str(d).lower() for d in descs)
        if "recgym" in j or "min-max normalized" in j or "dimensionless" in j:
            return False
        if any(k in j for k in ("gravity removed", "gravity-removed", "user acceleration",
                                "useracceleration", "linear acceleration")):
            return False
        if any(k in j for k in ("includes gravity", "including gravity", "with gravity",
                                "gravity included")):
            return True
    a = triad if isinstance(triad, np.ndarray) else triad.detach().cpu().numpy()
    a = a.astype(np.float64)
    m = a.mean(axis=0)
    dc = float(np.linalg.norm(m))
    rms = float(np.sqrt((a ** 2).sum(axis=1).mean())) + 1e-8
    axis_conc = float(np.max(np.abs(m))) / (dc + 1e-8)   # ->1 if one axis dominates
    return dc > 0.5 * rms and axis_conc > 0.6


def _mark_gravity_removed(desc: str) -> str:
    """Rewrite a channel description to state gravity was removed, stripping any
    contradictory 'includes gravity' clause first (avoids 'includes gravity
    (gravity removed)')."""
    import re
    d = re.sub(r"\([^)]*includes gravity[^)]*\)", "", desc, flags=re.I)
    # strip a "; includes gravity" / "includes gravity" clause with any leading separator, so the
    # sibling channel_description() format ("...; includes gravity") does not leave a dangling ";".
    d = re.sub(r"\s*;?\s*\bincludes gravity\b", "", d, flags=re.I)
    d = re.sub(r"\s{2,}", " ", d).strip().rstrip(",;").strip()
    if "gravity removed" not in d.lower():
        d = f"{d} (gravity removed)"
    return d


def _random_so3() -> "torch.Tensor":
    """Uniform-random rotation matrix (3x3, float32) from SO(3) (Haar measure).

    Sampled via a random unit quaternion (Marsaglia): four i.i.d. N(0,1), normalized,
    mapped to a rotation matrix. Uniform over orientations and always a proper rotation
    (det=+1, no reflection)."""
    q = np.random.randn(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    return torch.from_numpy(R)


class IMUAugmenter:
    """Applies the enabled augmentations (in AugmentationConfig.ORDER) to an
    IMUSample. Operates per sample on (T, C) tensors — padding is added later in
    collate, so no attention mask is needed here."""

    def __init__(self, config: "AugmentationConfig"):
        self.cfg = config

    def __call__(self, sample: "IMUSample") -> "IMUSample":
        for name in AugmentationConfig.ORDER:
            spec = getattr(self.cfg, name)
            if not spec.enabled or _random.random() >= spec.p:
                continue
            sample = getattr(self, "_" + name)(sample, spec)
        return sample

    # ---------- triad helper ----------
    @staticmethod
    def _triads(channel_names):
        """Return {location: [(indices3, group_name), ...]} for x/y/z triads only."""
        from data.scripts.curate.channels import (
            group_channels_by_sensor,
        )
        groups = group_channels_by_sensor(channel_names)
        ch_to_idx = {n: i for i, n in enumerate(channel_names)}
        def location(g):
            # Location = group name with its sensor-type token removed, robust to alternate
            # spellings ('watch_accel') and numeric range suffixes ('chest_acc16'). This keeps a
            # location's accel + gyro (+ mag) in ONE bucket so _rotation_3d rotates them by a
            # SHARED R (one rigid-body frame); mis-locating rotates accel apart from its gyro (#88).
            m = _SENSOR_TOKEN_RE.search(g)
            if not m:
                return g
            return (g[:m.start()] + g[m.end():]).strip("_")

        out = {}
        for g, chans in groups.items():
            if len(chans) != 3:
                continue
            out.setdefault(location(g), []).append(([ch_to_idx[c] for c in chans], g))
        return out

    # ---------- value-space (ported) ----------
    def _jitter(self, s, spec):
        scale = s.data.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        s.data = s.data + torch.randn_like(s.data) * (spec.sigma * scale)
        return s

    def _scale(self, s, spec):
        C = s.data.shape[1]
        factors = torch.empty(1, C, device=s.data.device).uniform_(spec.low, spec.high)
        s.data = s.data * factors
        return s

    def _time_shift(self, s, spec):
        T = s.data.shape[0]
        max_shift = max(1, int(T * spec.max_ratio))
        shift = int(np.random.randint(-max_shift, max_shift + 1))
        if shift > 0:
            fill = s.data[:shift].mean(0, keepdim=True).repeat(shift, 1)
            s.data = torch.cat([fill, s.data[:-shift]], 0)
        elif shift < 0:
            fill = s.data[shift:].mean(0, keepdim=True).repeat(-shift, 1)
            s.data = torch.cat([s.data[-shift:], fill], 0)
        return s

    def _time_warp(self, s, spec):
        T, C = s.data.shape
        if T < 10:
            return s
        orig = np.linspace(0, 1, T)
        knots = np.linspace(0, 1, spec.n_knots)
        vals = np.clip(knots + np.random.randn(spec.n_knots) * spec.strength, 0, 1)
        vals[0], vals[-1] = 0, 1
        vals = np.sort(vals)
        # PCHIP (monotone cubic), NOT interp1d cubic: an unconstrained cubic through the
        # knots overshoots past [0,1]; the clip then SATURATES warped_t, which edge-holds
        # the signal into a flat tail (dead ~25% of the window on strong draws) and can
        # even locally reverse time between knots. PCHIP is monotone + bounded by
        # construction -> a true time reparametrization. (Caught by M2 visual inspection.)
        warped_t = np.clip(interpolate.PchipInterpolator(knots, vals)(orig), 0, 1)
        x = s.data.detach().cpu().numpy()
        out = np.stack(
            [interpolate.interp1d(orig, x[:, c], kind="linear", fill_value="extrapolate")(warped_t)
             for c in range(C)],
            axis=-1,
        )
        s.data = torch.from_numpy(np.ascontiguousarray(out)).float().to(s.data.device)
        return s

    def _magnitude_warp(self, s, spec):
        T, C = s.data.shape
        if T < 10:
            return s
        grid = np.linspace(0, 1, T)
        knots = np.linspace(0, 1, spec.n_knots)
        x = s.data.detach().cpu().numpy().copy()
        for c in range(C):
            facs = np.clip(1.0 + np.random.randn(spec.n_knots) * spec.strength, 0.5, 1.5)
            curve = interpolate.interp1d(knots, facs, kind="cubic", fill_value="extrapolate")(grid)
            x[:, c] = x[:, c] * curve
        s.data = torch.from_numpy(x).float().to(s.data.device)
        return s

    # ---------- P1: gravity add/remove ----------
    def _gravity(self, s, spec):
        sr = float(s.sampling_rate)
        wn = spec.cutoff_hz / (sr / 2.0)
        if not (0.0 < wn < 1.0):     # cutoff above Nyquist (very low rate) -> skip
            return s
        T = s.data.shape[0]
        if T <= 3 * (spec.order + 1):   # filtfilt needs enough samples
            return s
        b, a = _sps.butter(spec.order, wn, btype="low")
        x = s.data.detach().cpu().numpy().astype(np.float64)
        desc = list(s.channel_descriptions)
        changed = False
        affected_sensor_ids = set()
        for _loc, triads in self._triads(s.channel_names).items():
            for idxs, gname in triads:
                if "acc" not in gname:       # only accelerometer carries gravity
                    continue
                if s.channel_mask is not None and not all(s.channel_mask[j] for j in idxs):
                    continue                 # canonical zero-padding is not a physical accelerometer
                if not _gravity_present(x[:, idxs], [desc[j] for j in idxs]):  # already gravity-removed -> skip
                    continue
                for j in idxs:
                    grav = _sps.filtfilt(b, a, x[:, j])
                    x[:, j] = x[:, j] - grav
                    desc[j] = _mark_gravity_removed(desc[j])
                    if s.sensor_id is not None:
                        affected_sensor_ids.add(s.sensor_id[j])
                changed = True
        if changed:
            s.data = torch.from_numpy(x).float().to(s.data.device)
            s.channel_descriptions = desc
            s.gravity_state = "removed"
            if s.sensor_descriptions is not None:
                affected = (affected_sensor_ids if s.sensor_id is not None
                            else set(range(len(s.sensor_descriptions))))
                sensor_desc = list(s.sensor_descriptions)
                for sid in affected:
                    sensor_desc[sid] = _mark_gravity_removed(sensor_desc[sid])
                s.sensor_descriptions = sensor_desc
        return s

    # ---------- P2: full uniform-random SO(3) rotation ----------
    def _rotation_3d(self, s, spec):
        """Rotate every co-located sensor triad by one shared uniform-random SO(3) R
        per body location (acc/gyro/mag at the same place rotate together, preserving
        their physical relationship). The gravity DC rotates with the accel signal, so
        this teaches gravity-direction invariance rather than scrambling an unseen cue."""
        triloc = self._triads(s.channel_names)
        if not triloc:
            return s
        x = s.data

        def loc_has_gravity(triads):
            for idxs, gname in triads:
                if "acc" in gname:
                    if s.channel_mask is not None and not all(s.channel_mask[j] for j in idxs):
                        continue
                    tri = x[:, idxs]
                    if _gravity_present(tri.detach().cpu().numpy(),
                                        [s.channel_descriptions[k] for k in idxs]):
                        return True
            return False

        for _loc, triads in triloc.items():
            # Only rotate locations whose accel still carries gravity — rotating
            # normalized / gravity-removed data would be a meaningless mixing of axes.
            if spec.require_gravity and not loc_has_gravity(triads):
                continue
            R = _random_so3().to(x.dtype).to(x.device)
            for idxs, _gname in triads:
                x[:, idxs] = torch.einsum("ij,tj->ti", R, x[:, idxs])
        s.data = x
        return s

    # ---------- P3: anti-aliased rate resample ----------
    def _rate(self, s, spec):
        old = float(s.sampling_rate)
        new = float(np.random.uniform(spec.min_hz, spec.max_hz))
        if old <= 0 or abs(new - old) < 1e-3:
            return s
        frac = Fraction(new / old).limit_denominator(50)
        up, down = frac.numerator, frac.denominator
        if up < 1 or down < 1:
            return s
        T = s.data.shape[0]
        if int(round(T * up / down)) < spec.min_samples:
            return s
        x = s.data.detach().cpu().numpy()
        y = _sps.resample_poly(x, up, down, axis=0)     # polyphase, anti-aliased
        s.data = torch.from_numpy(np.ascontiguousarray(y)).float().to(s.data.device)
        s.sampling_rate = old * up / down               # actual achieved rate
        return s

    # ---------- P5: random temporal crop (variable observation length) ----------
    def _window_crop(self, s, spec):
        T = s.data.shape[0]
        floor = min(T, spec.min_samples)
        lo = max(floor, int(round(spec.min_frac * T)))
        if lo >= T:                                        # nothing to crop (already at/below floor)
            return s
        length = int(np.random.randint(lo, T + 1))         # keep [lo, T] contiguous samples
        start = int(np.random.randint(0, T - length + 1))
        s.data = s.data[start:start + length].contiguous()
        return s

    # ---------- P4: channel / sensor-group dropout ----------
    def _channel_dropout(self, s, spec):
        from data.scripts.curate.channels import (
            group_channels_by_sensor,
        )
        names = s.channel_names
        drop = {i for i, n in enumerate(names) if any(g in n for g in spec.groups)}
        keep = [i for i in range(len(names)) if i not in drop]
        if not drop or len(keep) < 3:
            return s
        kept_names = [names[i] for i in keep]
        # require at least one full x/y/z triad to survive the drop
        if not any(len(v) == 3 for v in group_channels_by_sensor(kept_names).values()):
            return s
        s.data = s.data[:, keep]
        s.channel_names = kept_names
        s.channel_descriptions = [s.channel_descriptions[i] for i in keep]
        if s.role_descriptions is not None:
            s.role_descriptions = [s.role_descriptions[i] for i in keep]
        if s.channel_mask is not None:
            s.channel_mask = [s.channel_mask[i] for i in keep]
        # Factored sensors: accel and gyro are separate modality-level SENSORS, so dropping a channel
        # group REMOVES that modality's sensor entirely (rather than rewriting a modality phrase in a
        # single shared description). Keep only sensors that still own a surviving channel and compact
        # sensor_id to a dense [0..n) index into the pruned sensor_descriptions.
        if s.sensor_id is not None:
            kept_ids = [s.sensor_id[i] for i in keep]
            if s.sensor_descriptions is not None:
                used = sorted(set(kept_ids))
                remap = {old: new for new, old in enumerate(used)}
                s.sensor_descriptions = [s.sensor_descriptions[sid] for sid in used]
                s.sensor_id = [remap[sid] for sid in kept_ids]
            else:
                s.sensor_id = kept_ids
        return s

    # ---------- text: channel-description phrase paraphrase ----------
    def _channel_text_phrase(self, s, spec):
        # Paraphrase each channel description independently (surface form only; placement /
        # units / gravity are preserved by construction — see _paraphrase_channel).
        s.channel_descriptions = [_paraphrase_channel(d) for d in s.channel_descriptions]
        if s.role_descriptions is not None:
            s.role_descriptions = [_paraphrase_channel(d) for d in s.role_descriptions]
        if s.sensor_descriptions is not None:
            s.sensor_descriptions = [_paraphrase_channel(d) for d in s.sensor_descriptions]
        return s

    # ---------- text: channel-description dropout (neutralize, keep signal) ----------
    def _channel_text_dropout(self, s, spec):
        # Neutralize only REAL channels and never all of them (F10b): padded/absent channels are
        # masked out by the encoder, so neutralizing their text is a no-op — and counting them in
        # the budget let `max_frac` consume every real placement description on an acc-only stream
        # (3 real of 6 slots). Bound over the REAL-channel count and always keep >=1 real described.
        cm = s.channel_mask
        real = [i for i in range(len(s.channel_descriptions))
                if cm is None or (i < len(cm) and cm[i])]
        if len(real) > 1:
            max_drop = max(1, int(spec.max_frac * len(real)))
            k = min(_random.randint(1, max_drop), len(real) - 1)   # keep >=1 real channel described
            dropped = _random.sample(real, k)
            desc = list(s.channel_descriptions)
            for i in dropped:
                desc[i] = spec.neutral
            s.channel_descriptions = desc

            # The same selected CHANNEL roles are neutralized in the factored path. Sensor identity
            # is shared by every channel on that sensor, so dropping a one-sensor description would
            # erase 100% of placement/gravity metadata and violate max_frac.
            if s.role_descriptions is not None:
                role_desc = list(s.role_descriptions)
                for i in dropped:
                    role_desc[i] = spec.neutral
                s.role_descriptions = role_desc
        return s

    # ---------- text: label paraphrase (dataset-specific synonyms + templates) ----------
    def _label_text(self, s, spec):
        from data.scripts.labels.label_augmentation import augment_label
        # Outer `p` already decided we augment; call augment_label unconditionally (rate=1.0).
        s.label_text = augment_label(
            s.label, s.dataset_name, augmentation_rate=1.0,
            use_synonyms=spec.use_synonyms, use_templates=spec.use_templates,
        )
        return s


