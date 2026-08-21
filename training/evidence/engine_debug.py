"""Debug sweep for the compact evidence engine: encoder -> rows -> retrieve -> mix -> vote.

WHAT THIS IS
------------
This is the synthetic mechanism-level companion to the real-corpus trainer. It runs the same full
learned path — temporal ``SetTokenizerEncoder``, ``live_sensor_rows``, ``EvidenceEngine``, and loss
— on a corpus built so each mechanism has an independently legible signal:

  * 8 concept GROUPS x 3 variants = 24 labels. Label-text vectors cluster by group; signal
    frequency clusters by group with per-variant modulation. So the ConSE text bridge (k=0),
    retrieval (signal similarity), and enrollment (support binding) all carry real signal, and a
    broken mechanism shows up as a specific cell failing rather than a vague underperformance.
  * Two acquisition configs (phone with gravity, watch gravity-removed), two modalities per
    window, per-subject frequency jitter — enough heterogeneity to exercise the descriptor path
    and the physics-violation diagnostic without turning the harness into a data loader.

WHAT IT CHECKS
--------------
  1. Activation magnitudes at every stage of the network, at init and after training (NaN/Inf
     anywhere is a hard failure).
  2. Brief real training (AdamW, the intended two learning rates): loss falls, accuracy beats
     chance, every learnable module receives gradient, update ratios are in a sane band.
  3. A gradient audit in the style of the 2026-08 sweep: per-module gradient norm, update ratio,
     and ANSWER SPECIFICITY — 1 - cos(grad, grad with targets rolled). ~0 means the module's
     gradient does not depend on which answer is correct, the signature that exposed the
     answer-blind gate.
  4. An edge-case battery: candidate counts 2 and 16, k=0-only, fully-enrolled memory, a bank
     smaller than top_k, one-recording banks, determinism under a fixed generator, and the
     "semantic" readout through all of it.

Run:  python -m training.evidence.engine_debug [--steps 300] [--device cuda]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.blocks import AttentionSpec
from model.evidence.engine import EngineConfig, EvidenceEngine
from model.evidence.evidence_mixer import EvidenceMixerConfig
from model.tokenizer.encoder import SetTokenizerEncoder
from training.tokenizer.episodic import live_sensor_rows

# ---------------------------------------------------------------------------- synthetic corpus
N_GROUPS, VARIANTS = 8, 3
N_LABELS = N_GROUPS * VARIANTS
RATE_HZ, WINDOW_S, PATCH_S = 50.0, 6.0, 1.0
P = int(WINDOW_S / PATCH_S)
SAMPLES = int(RATE_HZ * PATCH_S)
DFT_SIZE = 256                      # the filterbank's fixed frame; patches are zero-padded to it

CONFIGS = [
    {"accel": "a phone accelerometer in the front trouser pocket; includes gravity; "
              "recorded alongside a gyroscope",
     "gyro": "a phone gyroscope in the front trouser pocket; recorded alongside an accelerometer",
     "gravity": "present"},
    {"accel": "a watch accelerometer on the left wrist; gravity removed; "
              "recorded alongside a gyroscope",
     "gyro": "a watch gyroscope on the left wrist; recorded alongside an accelerometer",
     "gravity": "removed"},
]


def make_label_space(rng: np.random.Generator) -> torch.Tensor:
    """(V, 384) unit vectors, clustered by concept group — the frozen 'SBERT' of this corpus."""
    centroids = rng.standard_normal((N_GROUPS, 384))
    out = []
    for group in range(N_GROUPS):
        for _ in range(VARIANTS):
            v = centroids[group] + 0.35 * rng.standard_normal(384)
            out.append(v / np.linalg.norm(v))
    return torch.tensor(np.stack(out), dtype=torch.float32)


def synth_window(label: int, subject: int, config: int, rng: np.random.Generator) -> np.ndarray:
    """(P, SAMPLES, 6) — group sets the base frequency, variant modulates, subject jitters."""
    group, variant = divmod(label, VARIANTS)
    freq = (0.8 + 0.9 * group) * (1.0 + 0.08 * (variant - 1)) * (1.0 + 0.02 * ((subject % 7) - 3))
    accel_amp = 0.6 + 0.15 * group
    gyro_amp = 1.4 - 0.12 * group
    t = np.arange(int(RATE_HZ * WINDOW_S)) / RATE_HZ
    phase = rng.uniform(0, 2 * np.pi)
    base = np.sin(2 * np.pi * freq * t + phase)
    harmonic = 0.3 * np.sin(2 * np.pi * 2 * freq * t + phase * 1.7)
    window = np.zeros((len(t), 6), dtype=np.float32)
    for axis in range(3):
        scale = (0.5, 1.0, 0.75)[axis]
        window[:, axis] = accel_amp * scale * (base + harmonic)
        window[:, 3 + axis] = gyro_amp * scale * np.cos(2 * np.pi * freq * t + phase + axis)
    if CONFIGS[config]["gravity"] == "present":
        window[:, 2] += 1.0                              # gravity DC on accel z
    window += 0.05 * rng.standard_normal(window.shape).astype(np.float32)
    patches = window.reshape(P, SAMPLES, 6)
    padded = np.zeros((P, DFT_SIZE, 6), dtype=np.float32)
    padded[:, :SAMPLES] = patches                    # filterbank contract: zero-pad to DFT frame
    return padded


# ---------------------------------------------------------------------------- episode assembly
def build_episode(rng, *, n_candidates, queries_per_candidate=2, max_support=3, bank_windows=96,
                  force_support=None):
    """One episode as (windows, labels, enrolled, configs, target_slot_per_query_window, C)."""
    candidates = rng.choice(N_LABELS, n_candidates, replace=False)
    windows, labels, enrolled, configs = [], [], [], []
    targets = []
    for slot, label in enumerate(candidates):
        subject = int(rng.integers(0, 5))
        for _ in range(queries_per_candidate):
            config = int(rng.integers(0, 2))
            windows.append(synth_window(int(label), subject, config, rng))
            labels.append(int(label)); enrolled.append(-1); configs.append(config)
            targets.append(slot)
    support_k = int(rng.integers(0, max_support + 1)) if force_support is None else force_support
    if support_k:
        enrolled_slots = rng.choice(n_candidates, int(rng.integers(1, n_candidates + 1)),
                                    replace=False)
        for slot in enrolled_slots:
            for _ in range(support_k):
                config = int(rng.integers(0, 2))
                windows.append(synth_window(int(candidates[slot]), int(rng.integers(5, 10)),
                                            config, rng))
                labels.append(int(candidates[slot])); enrolled.append(int(slot))
                configs.append(config)
    outside = np.setdiff1d(np.arange(N_LABELS), candidates)
    for _ in range(bank_windows):
        label = int(rng.choice(outside))
        config = int(rng.integers(0, 2))
        windows.append(synth_window(label, int(rng.integers(0, 10)), config, rng))
        labels.append(label); enrolled.append(-1); configs.append(config)
    n_query = n_candidates * queries_per_candidate
    return (np.stack(windows), np.array(labels), np.array(enrolled), np.array(configs),
            np.array(targets), candidates, n_query)


def encode_episode(encoder, ep, device):
    windows, labels, enrolled, configs, targets, candidates, n_query = ep
    B = len(windows)
    batch = {
        "patches": torch.tensor(windows, device=device),
        "rates": torch.full((B,), RATE_HZ, device=device),
        "patch_len": torch.full((B, P), SAMPLES, dtype=torch.long, device=device),
        "role_texts": [["x", "y", "z", "x", "y", "z"]] * B,
        "positions": torch.arange(P, device=device, dtype=torch.float32)
                          .unsqueeze(0).expand(B, P).contiguous(),
        "patch_durations": torch.full((B, P), PATCH_S, device=device),
        "channel_mask": torch.ones(B, 6, dtype=torch.bool, device=device),
        "patch_padding_mask": torch.ones(B, P, dtype=torch.bool, device=device),
        "sensor_texts": [[CONFIGS[c]["accel"], CONFIGS[c]["gyro"]] for c in configs],
        "sensor_id": torch.tensor([[0, 0, 0, 1, 1, 1]] * B, device=device),
        "source_rates": torch.full((B,), RATE_HZ, device=device),
        "gravity_state": [CONFIGS[c]["gravity"] for c in configs],
    }
    out = encoder(
        batch["patches"], batch["rates"], batch["patch_len"], batch["role_texts"],
        batch["positions"], patch_durations=batch["patch_durations"],
        channel_mask=batch["channel_mask"], patch_padding_mask=batch["patch_padding_mask"],
        sensor_texts=batch["sensor_texts"], sensor_id=batch["sensor_id"],
        source_rate_hz=batch["source_rates"], return_retrieval_tokens=True,
    )
    labels_t = torch.tensor(labels, device=device)
    enrolled_t = torch.tensor(enrolled, device=device)
    query = live_sensor_rows(out, batch, labels=labels_t, enrolled_candidate=enrolled_t,
                             select=torch.arange(n_query, device=device))
    memory = live_sensor_rows(out, batch, labels=labels_t, enrolled_candidate=enrolled_t,
                              select=torch.arange(n_query, B, device=device))
    return query, memory, torch.tensor(candidates, device=device), \
        torch.tensor(targets, device=device), n_query


def episode_loss(engine, label_text, query, memory, candidates, targets, n_query, *,
                 generator=None, collect_stats=False):
    candidate_text = label_text[candidates]
    result = engine(query.rows, memory.rows, candidate_text, label_text,
                    generator=generator, collect_stats=collect_stats)
    merged = result["logits"].new_zeros((n_query, len(candidates)))
    merged = merged.index_add(0, query.window, result["logits"])
    loss = F.cross_entropy((merged + 1e-8).log(), targets)
    accuracy = float((merged.argmax(1) == targets).float().mean())
    return loss, accuracy, result


# ---------------------------------------------------------------------------- instrumentation
def tensor_stats(value):
    tensors = []
    if torch.is_tensor(value):
        tensors = [value]
    elif isinstance(value, dict):
        tensors = [v for v in value.values() if torch.is_tensor(v) and v.is_floating_point()]
    elif isinstance(value, (tuple, list)):
        tensors = [v for v in value if torch.is_tensor(v) and v.is_floating_point()]
    tensors = [t for t in tensors if t.is_floating_point() and t.numel()]
    if not tensors:
        return None
    flat = torch.cat([t.detach().reshape(-1) for t in tensors])
    finite = torch.isfinite(flat)
    return {"absmean": float(flat[finite].abs().mean()) if finite.any() else float("nan"),
            "std": float(flat[finite].std()) if finite.sum() > 1 else 0.0,
            "absmax": float(flat[finite].abs().max()) if finite.any() else float("nan"),
            "nonfinite": int((~finite).sum())}


class ActivationMonitor:
    """Forward hooks over the named stages of the network."""

    def __init__(self, encoder, engine):
        self.records, self.handles = {}, []
        stages = {
            "encoder.filterbank": encoder.filterbank,
            "encoder.sensor_fold": encoder.sensor_fold,
            "encoder.descriptor_proj": encoder.descriptor_proj,
            "encoder.trunk": encoder.transformer,
            "scorer": engine.scorer,
        }
        # The fixed-cosine scorer has no learned submodules and the mixer can be absent entirely;
        # monitor what the configuration actually built rather than assuming the learned stack.
        if engine.cfg.scorer.learned:
            stages["scorer.query_proj"] = engine.scorer.query_proj
            stages["scorer.memory_proj"] = engine.scorer.memory_proj
        if engine.mixer is not None:
            stages["mixer.attention_stack"] = engine.mixer.stack
            stages["mixer"] = engine.mixer
        for layer_index, norm in enumerate(encoder.transformer.attn):
            stages[f"encoder.trunk.attn[{layer_index}]"] = norm
        for name, module in stages.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def record(_module, _inputs, output):
            stats = tensor_stats(output)
            if stats is not None:
                self.records[name] = stats
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()

    def table(self) -> str:
        rows = [f"{'stage':32s} {'|out| mean':>10s} {'std':>9s} {'|out| max':>10s} {'nonfinite':>9s}"]
        rows.append("-" * 74)
        for name, s in self.records.items():
            rows.append(f"{name:32s} {s['absmean']:>10.4f} {s['std']:>9.4f} "
                        f"{s['absmax']:>10.3f} {s['nonfinite']:>9d}")
        return "\n".join(rows)


def module_groups(encoder, engine):
    groups = defaultdict(list)
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            groups["encoder." + name.split(".")[0]].append(param)
    for part in ("scorer", "mixer"):
        for name, param in getattr(engine, part).named_parameters():
            if param.requires_grad:
                key = f"{part}.{name.split('.')[0]}"
                groups[key].append(param)
    return dict(groups)


def collect_grads(groups):
    return {name: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                             for p in params]).clone()
            for name, params in groups.items()}


def gradient_audit(encoder, engine, label_text, episodes, groups, device, title):
    """Grad norm + answer specificity per module, on a fixed set of episodes."""
    def grads_with(wrong_answers: bool):
        encoder.zero_grad(set_to_none=True); engine.zero_grad(set_to_none=True)
        total = 0.0
        for ep in episodes:
            query, memory, candidates, targets, n_query = encode_episode(encoder, ep, device)
            answer = (targets + 1) % len(candidates) if wrong_answers else targets
            loss, _, _ = episode_loss(engine, label_text, query, memory, candidates,
                                      answer, n_query,
                                      generator=torch.Generator(device="cpu").manual_seed(0))
            total = total + loss
        total.backward()
        return collect_grads(groups)

    true_grads = grads_with(False)
    rolled_grads = grads_with(True)
    lines = [f"\n=== gradient audit: {title} ===",
             f"{'module':28s} {'params':>10s} {'grad norm':>12s} {'answer spec':>12s}"]
    lines.append("-" * 66)
    problems = []
    # Parameters with no path to THIS loss by design, not by accident. mask_token exists for
    # Phase-A JEPA masking; Phase B never masks a token, so a zero gradient here is the module
    # working as documented — the real trainer should simply not hand it to the optimizer.
    expected_dead = {"encoder.mask_token"}
    # With the fixed-cosine default the scorer registers no parameters, so no scorer group exists;
    # nothing to whitelist. (If the learned scorer is enabled its groups must all be live.)
    for name, params in groups.items():
        count = sum(p.numel() for p in params)
        g, r = true_grads[name], rolled_grads[name]
        norm = float(g.norm())
        if norm == 0.0:
            if name not in expected_dead:
                problems.append(f"DEAD: {name} received zero gradient")
            spec = float("nan")
        else:
            spec = 1.0 - float(F.cosine_similarity(g, r, dim=0))
        tag = ("  dead by design (Phase-A only)" if norm == 0 and name in expected_dead
               else ("  DEAD" if norm == 0 else ""))
        lines.append(f"{name:28s} {count:>10,} {norm:>12.3e} {spec:>12.3f}" + tag)
    encoder.zero_grad(set_to_none=True); engine.zero_grad(set_to_none=True)
    return "\n".join(lines), problems


# ---------------------------------------------------------------------------- training smoke test
def train_smoke(readout, steps, device, log, seed=0, **engine_overrides):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    label_text = make_label_space(np.random.default_rng(123)).to(device)

    spec = AttentionSpec()
    encoder = SetTokenizerEncoder(
        d_model=spec.d_model, num_layers=3, num_heads=spec.n_heads,
        dim_feedforward=spec.dim_feedforward, dropout=spec.dropout,
        token_granularity="sensor", text_conditioning="factored",
        trunk="temporal", descriptor_prediction=False,
    ).to(device)
    engine = EvidenceEngine(None, EngineConfig(
        mixer=EvidenceMixerConfig(readout=readout), **engine_overrides)).to(device)

    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": list(engine.parameters()), "lr": 3e-4},
    ], weight_decay=0.01)
    groups = module_groups(encoder, engine)

    # -- init-time activation table + gradient audit on fixed episodes
    audit_episodes = [build_episode(np.random.default_rng(7 + i), n_candidates=6)
                      for i in range(2)]
    monitor = ActivationMonitor(encoder, engine)
    encoder.train(); engine.train()
    query, memory, candidates, targets, n_query = encode_episode(
        encoder, audit_episodes[0], device)
    _, _, first = episode_loss(engine, label_text, query, memory, candidates, targets, n_query,
                               collect_stats=True)
    log(f"\n=== activation magnitudes at init ({readout}) ===")
    log(monitor.table())
    monitor.close()
    log("\nengine stats at init: " + "  ".join(
        f"{k.split('/')[-1]}={v:.3f}" for k, v in sorted(first["stats"].items())))
    audit_text, audit_problems = gradient_audit(
        encoder, engine, label_text, audit_episodes, groups, device, f"{readout} @ init")
    log(audit_text)

    # -- training
    log(f"\n=== training {steps} steps ({readout}, "
        f"{', '.join(f'{k}={v}' for k, v in engine_overrides.items()) or 'defaults'}) ===")
    history, problems = [], list(audit_problems)
    start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss, step_acc = 0.0, []
        for _ in range(2):                                            # 2 episodes per step
            ep = build_episode(rng, n_candidates=int(rng.choice([4, 6, 8])))
            query, memory, candidates, targets, n_query = encode_episode(encoder, ep, device)
            loss, accuracy, _ = episode_loss(engine, label_text, query, memory, candidates,
                                             targets, n_query)
            (loss / 2).backward()
            step_loss += float(loss.detach()) / 2; step_acc.append(accuracy)
        if not math.isfinite(step_loss):
            problems.append(f"NON-FINITE LOSS at step {step}")
            break
        before = {name: torch.cat([p.detach().reshape(-1) for p in params]).clone()
                  for name, params in groups.items()} if step in (0, steps - 1) else None
        torch.nn.utils.clip_grad_norm_(
            [p for group in groups.values() for p in group], 5.0)
        optimizer.step()
        history.append((step_loss, float(np.mean(step_acc))))
        if before is not None:
            log(f"\n  update ratios |Δp|/|p| at step {step}:")
            for name, params in groups.items():
                after = torch.cat([p.detach().reshape(-1) for p in params])
                delta = float((after - before[name]).norm())
                base = float(before[name].norm())
                if base < 1e-6:
                    # Zero-initialised parameters (ScaledSum gains) have no norm to be a fraction
                    # of; the ratio would divide by nothing and read as an explosion.
                    log(f"    {name:28s} |Δp| {delta:.2e} (param initialised at zero)")
                    continue
                ratio = delta / base
                flag = ("  <-- LARGE" if ratio > 0.05 else
                        ("  <-- inert (Phase-A only)" if ratio < 1e-9
                         and name == "encoder.mask_token" else
                         ("  <-- inert" if ratio < 1e-9 else "")))
                log(f"    {name:28s} {ratio:.2e}{flag}")
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            recent = history[-20:]
            gains = engine.mixer.telemetry() if engine.mixer is not None else {}
            if engine.cfg.scorer.learned:
                gains["retrieval/base_gain"] = float(engine.scorer.base_gain)
                gains["retrieval/residual_gain"] = float(engine.scorer.residual_gain)
            log(f"  step {step:>4d}  loss {np.mean([h[0] for h in recent]):.4f}  "
                f"acc {np.mean([h[1] for h in recent]):.3f}  "
                + "  ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in sorted(gains.items())))
    seconds = time.perf_counter() - start
    log(f"  {seconds:.1f} s total, {seconds / max(len(history), 1) * 1000:.0f} ms/step "
        f"(2 episodes = ~230 windows encoded per step)")

    # -- end-of-training audit + activations
    monitor = ActivationMonitor(encoder, engine)
    query, memory, candidates, targets, n_query = encode_episode(
        encoder, audit_episodes[0], device)
    _, _, last = episode_loss(engine, label_text, query, memory, candidates, targets, n_query,
                              collect_stats=True)
    log(f"\n=== activation magnitudes after training ({readout}) ===")
    log(monitor.table()); monitor.close()
    log("\nengine stats after training: " + "  ".join(
        f"{k.split('/')[-1]}={v:.3f}" for k, v in sorted(last["stats"].items())))
    audit_text, late_problems = gradient_audit(
        encoder, engine, label_text, audit_episodes, groups, device, f"{readout} after training")
    log(audit_text); problems += late_problems

    early = np.mean([h[0] for h in history[:10]]); late = np.mean([h[0] for h in history[-10:]])
    early_acc = np.mean([h[1] for h in history[:10]]); late_acc = np.mean([h[1] for h in history[-10:]])
    log(f"\nloss {early:.3f} -> {late:.3f}   accuracy {early_acc:.3f} -> {late_acc:.3f} "
        f"(chance over C in {{4,6,8}} is ~0.18)")
    # The first and last ten-step windows overlap on a tiny mechanism smoke, so comparing them
    # would inevitably call a one-step run "no improvement". Trend assertions need disjoint data.
    if len(history) >= 20:
        if late >= early:
            problems.append(f"{readout}: loss did not fall ({early:.3f} -> {late:.3f})")
        if late_acc < 0.5:
            problems.append(f"{readout}: accuracy after training only {late_acc:.3f}")
    else:
        log("trend assertion skipped: fewer than 20 steps")
    for name, param in list(encoder.named_parameters()) + list(engine.named_parameters()):
        if not torch.isfinite(param).all():
            problems.append(f"NON-FINITE PARAMETER after training: {name}")
    return problems


# ---------------------------------------------------------------------------- edge battery
def edge_battery(device, log):
    log("\n=== edge-case battery ===")
    label_text = make_label_space(np.random.default_rng(123)).to(device)
    problems = []
    for readout, scope in (("weights", "topk"), ("weights", "bank"), ("semantic", "topk")):
        encoder = SetTokenizerEncoder(
            d_model=128, num_layers=3, num_heads=4, dim_feedforward=256, dropout=0.1,
            token_granularity="sensor", text_conditioning="factored",
            trunk="temporal", descriptor_prediction=False,
        ).to(device).eval()
        engine = EvidenceEngine(None, EngineConfig(
            vote_scope=scope,
            mixer=EvidenceMixerConfig(readout=readout))).to(device).eval()

        def run(tag, **kwargs):
            rng = np.random.default_rng(kwargs.pop("seed", 11))
            ep = build_episode(rng, **kwargs)
            with torch.no_grad():
                q, m, cand, tgt, nq = encode_episode(encoder, ep, device)
                loss, acc, result = episode_loss(
                    engine, label_text, q, m, cand, tgt, nq,
                    generator=torch.Generator(device="cpu").manual_seed(1))
            ok = bool(torch.isfinite(result["logits"]).all()) and math.isfinite(float(loss))
            log(f"  [{readout:8s}/{scope:4s}] {tag:38s} loss {float(loss):7.3f}  "
                f"logits finite {ok}  rows {len(m.rows.feature)}")
            if not ok:
                problems.append(f"{readout}/{scope}/{tag}: non-finite output")
            return result

        run("C=2 minimum candidates", n_candidates=2)
        run("C=16 maximum candidates", n_candidates=16)
        run("k=0 zero-shot only", n_candidates=6, force_support=0)
        run("heavy support (k=3, all slots eligible)", n_candidates=4, force_support=3)
        run("bank smaller than top_k (12 windows)", n_candidates=4, force_support=0,
            bank_windows=12)
        run("large bank (512 windows)", n_candidates=6, bank_windows=512)

        # determinism under a fixed generator
        rng = np.random.default_rng(5)
        ep = build_episode(rng, n_candidates=6)
        with torch.no_grad():
            q, m, cand, tgt, nq = encode_episode(encoder, ep, device)
            a = engine(q.rows, m.rows, label_text[cand], label_text,
                       generator=torch.Generator(device="cpu").manual_seed(3))["logits"]
            b = engine(q.rows, m.rows, label_text[cand], label_text,
                       generator=torch.Generator(device="cpu").manual_seed(3))["logits"]
        if not torch.allclose(a, b):
            problems.append(f"{readout}/{scope}: same generator seed produced different logits")
        log(f"  [{readout}/{scope}] {'determinism under a fixed generator':38s} "
            f"max |Δ| {float((a - b).abs().max()):.1e}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--semantic-steps", type=int, default=120)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    out_dir = Path("training/evidence/outputs/engine_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "sweep_report.txt"
    lines = []

    def log(message=""):
        print(message, flush=True)
        lines.append(str(message))

    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    problems = []
    # The three committed operating points: the default (cosine retrieval, top-k vote), the
    # bank-wide vote, and the semantic readout. Each must train, keep every parameter live, and
    # stay finite.
    problems += train_smoke("weights", args.steps, device, log)
    problems += train_smoke("weights", args.steps // 2, device, log, vote_scope="bank")
    problems += train_smoke("semantic", args.semantic_steps, device, log)
    problems += edge_battery(device, log)

    log("\n" + "=" * 74)
    if problems:
        log(f"PROBLEMS FOUND ({len(problems)}):")
        for problem in problems:
            log(f"  - {problem}")
    else:
        log("SWEEP CLEAN: no dead modules, no non-finite values, both readouts learn, "
            "all edge cases pass.")
    report.write_text("\n".join(lines))
    log(f"\nreport: {report}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
