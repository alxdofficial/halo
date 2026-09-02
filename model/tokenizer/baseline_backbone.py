"""Frozen pretrained backbones wearing HALO's Phase-B encoder contract.

Purpose: answer "has our encoder been the limiting factor all along?" by swapping harnet5 or
UniMTS in for `SetTokenizerEncoder` while EVERYTHING downstream -- memory bank, pair scoring,
top-k, evidence mixer, text vote -- stays byte-identical.

The engine consumes exactly two tensors from an encoder (see `live_sensor_rows`):
    retrieval_tokens  (B, P, N, d)   one row per (patch, sensor)
    sensor_present    (B, N)         which sensor slots are live
Everything else a row needs (modality, gravity convention, placement) is read off the batch, not
the encoder, so a faithful stand-in only has to produce those two.

Two deliberate choices, both documented because they shape what the comparison means:

1. GRANULARITY. harnet5 wants 5 s at 30 Hz and UniMTS wants 10 s at 20 Hz; our patches are 1 s.
   Running either on a 1 s slice is far outside its training distribution, so each backbone emits
   ONE row for the whole source window. The comparison arm applies the same duration-weighted
   window pooling to HALO. Duplicate copies never enter retrieval or voting.

2. MODALITY. Both published backbones are accelerometer-only. Gyroscope rows and gravity-removed
   accelerometer rows are therefore absent in EVERY arm of the comparison. Feeding rad/s through an
   accelerometer trunk, or feeding linear acceleration to a gravity-dependent pretrained model,
   would not be a representation comparison at all.

The projection to the engine's width is always trainable -- a frozen backbone whose output could
not be linearly re-based would be testing the projection, not the representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as audio_transforms


class _TrunkShim(nn.Module):
    """Publishes the layer count the engine reads; the backbone has no trunk of ours."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.num_layers = int(n_layers)


def _center_crop_or_wrap(x: torch.Tensor, length: int) -> torch.Tensor:
    """HARNet adapter rule: center crop long signals, symmetric wrap-pad short ones."""
    current = x.shape[1]
    if current >= length:
        start = (current - length) // 2
        return x[:, start:start + length]
    left = (length - current) // 2
    index = (torch.arange(length, device=x.device) - left).remainder(current)
    return x.index_select(1, index)


def _start_crop_or_wrap(x: torch.Tensor, length: int) -> torch.Tensor:
    """UniMTS adapter rule: truncate from the start or append wrap-padding on the right."""
    current = x.shape[1]
    if current >= length:
        return x[:, :length]
    index = torch.arange(length, device=x.device).remainder(current)
    return x.index_select(1, index)


def _stream_joint(stream_key: str) -> int:
    """Map a ``dataset/stream`` key to the placement joint used by the UniMTS adapter."""
    from baselines.unimts.adapter import (
        DEFAULT_JOINT,
        JOINT_BY_DS,
        _PLACEMENT_KEYWORDS,
        _SIDE_PLACEMENT_JOINTS,
    )

    dataset, _, stream = str(stream_key).partition("/")
    name = f"{dataset}/{stream}".lower()
    for keyword, joint in _SIDE_PLACEMENT_JOINTS:
        if keyword in name:
            return int(joint)
    for keyword, joint in _PLACEMENT_KEYWORDS:
        if keyword in name:
            return int(joint)
    return int(JOINT_BY_DS.get(dataset, DEFAULT_JOINT))


class BaselineRowEncoder(nn.Module):
    """A pretrained backbone presented to the engine as a sensor-granularity row encoder."""

    #: guards in `encode_batch` check these; the stand-in satisfies the same contract
    trunk = "temporal"
    token_granularity = "sensor"
    use_sensor_isolated_retrieval = True
    requires_stream_metadata = True

    def __init__(self, backbone: str, d_model: int = 128, freeze: bool = True,
                 device: torch.device | None = None):
        super().__init__()
        if backbone not in {"harnet", "unimts"}:
            raise ValueError("backbone must be 'harnet' or 'unimts'")
        self.backbone_name = backbone
        if not freeze:
            raise ValueError(
                "the encoder comparison keeps released third-party backbones frozen; "
                "only the common projection and evidence engine are trainable"
            )
        self.freeze = True
        self.retrieval_granularity = "window"
        device = device or torch.device("cpu")

        if backbone == "harnet":
            from baselines.harnet.adapter import _load_harnet
            module = _load_harnet(2, device)
            # harnet's feature extractor is the ResNet trunk with its classifier removed
            self.net = getattr(module, "feature_extractor", module)
            self.in_hz, self.in_len, self.out_dim = 30.0, 150, 512
        else:
            from baselines.unimts.adapter import N_JOINTS
            from baselines.unimts.adapter import UniMTSAdapter
            self._uni = UniMTSAdapter()
            loaded = self._uni.setup(device)["model"]
            # `encode_image` is exactly `model.acc(...).squeeze(-1).squeeze(-1)`. The surrounding
            # CLIP text tower is 63M frozen parameters and is never called because all three arms use
            # HALO's shared MiniLM label/sensor text path. Retain only the released ST-GCN.
            self.net = loaded.model.acc
            del loaded
            self.n_joints = N_JOINTS
            self.in_hz, self.in_len, self.out_dim = 20.0, 200, 512

        self.net.eval()
        self.net.requires_grad_(False)
        # UniMTS is a deep 22-joint ST-GCN and is dominated by small elementwise/kernel launches.
        # Inductor fuses those operations without changing the released graph. Keep the compiled
        # callable outside nn.Module registration so checkpoints retain the released ``net.*`` keys
        # and remain loadable on CPU or on machines where compilation is unavailable.
        compiled = None
        if backbone == "unimts" and device.type == "cuda" and hasattr(torch, "compile"):
            compiled = torch.compile(self.net, mode="reduce-overhead", dynamic=False)
        object.__setattr__(self, "_compiled_net", compiled)
        # Functional sinc resampling rebuilds its filter kernel on every call. These modules are a
        # runtime cache (deliberately not registered/checkpointed); each is created on the input's
        # actual device and dtype and is immutable thereafter.
        self._resamplers: dict[tuple[int, int, str, int | None, torch.dtype], nn.Module] = {}
        self.proj = nn.Linear(self.out_dim, d_model)
        # Our trunk emits rows through a final normalization, so every row it produces has the same
        # norm (sqrt(d)). The backbones do not: harnet's raw rows span a 27x range. Cosine
        # retrieval is scale-free, but the evidence mixer reads row FEATURES, so an arm whose token
        # magnitudes vary by 27x is being tested on a different numerical regime rather than on its
        # representation. Matching the row-scale contract removes that confound.
        self.row_norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.d_model = int(d_model)
        # The engine reads its own attention geometry off the encoder. Handing back the SAME spec
        # our encoder publishes keeps the scorer/mixer/vote byte-identical across arms, so the
        # comparison isolates the backbone and nothing else.
        from model.blocks import AttentionSpec
        from model.tokenizer.channel_text import TokenTextEncoder
        self.attention_spec = AttentionSpec(d_model=d_model, n_heads=4, ffn_mult=2, dropout=0.1)
        self.transformer = _TrunkShim(n_layers=3)
        # The sensor-description embedding is the TEXT side, not the backbone: `live_sensor_rows`
        # reads it off every row, and holding it identical across arms is what makes the swap a
        # test of the signal encoder alone.
        self.text_encoder = TokenTextEncoder()

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode no matter what the trainer does to the parent.

        `nn.Module.train()` recurses, so the trainer's `encoder.train()` would otherwise flip
        harnet's BatchNorm layers back on and let their running statistics drift on our corpus --
        a frozen arm that is not actually frozen, and the drift would be invisible in the loss.
        """
        super().train(mode)
        if self.freeze:
            self.net.eval()
        self.text_encoder.eval()
        return self

    # ---------------------------------------------------------------- backbone feature call
    def _features(self, window: torch.Tensor, joint: torch.Tensor | None,
                  *, compiled: bool = False) -> torch.Tensor:
        """(M, required_length, 3) acceleration in g -> (M, out_dim)."""
        x = window
        if self.backbone_name == "harnet":
            return self.net(x.permute(0, 2, 1)).flatten(1)          # (M, 3, 150) -> (M, 1024)
        # UniMTS: place the 3 channels at one SMPL joint, zeros elsewhere -> (M,3,200,22,1)
        M, T, _ = x.shape
        grid = x.new_zeros((M, T, self.n_joints, 3))
        idx = joint if joint is not None else torch.zeros(M, dtype=torch.long, device=x.device)
        grid[torch.arange(M, device=x.device), :, idx, :] = x * 9.80665   # g -> m/s^2
        runner = self._compiled_net if compiled and self._compiled_net is not None else self.net
        return runner(grid.permute(0, 3, 1, 2).unsqueeze(-1)).squeeze(-1).squeeze(-1)

    def _resample(self, x: torch.Tensor, orig_hz: float) -> torch.Tensor:
        if abs(float(orig_hz) - self.in_hz) < 1e-6:
            return x
        scale = 1000
        orig = int(round(float(orig_hz) * scale))
        target = int(round(self.in_hz * scale))
        key = (orig, target, x.device.type, x.device.index, x.dtype)
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = audio_transforms.Resample(orig, target, dtype=x.dtype).to(x.device)
            self._resamplers[key] = resampler
        return resampler(x.transpose(1, 2).contiguous()).transpose(1, 2)

    # ---------------------------------------------------------------- the encoder contract
    def forward(self, patches, rates, patch_len, role_texts, positions, *,
                patch_durations=None, channel_mask=None, patch_padding_mask=None,
                sensor_texts=None, sensor_id=None, source_rate_hz=None,
                return_retrieval_tokens=True, streams=None, gravity_state=None, **_ignored):
        B, P, S, C = patches.shape
        device = patches.device
        if sensor_id is None or channel_mask is None or patch_padding_mask is None:
            raise ValueError("BaselineRowEncoder needs sensor_id, channel_mask, patch_padding_mask")
        if not sensor_texts:
            raise ValueError("BaselineRowEncoder needs sensor_texts (one description per sensor)")
        if streams is None or len(streams) != B:
            raise ValueError("BaselineRowEncoder needs one authoritative stream key per window")
        if gravity_state is None or len(gravity_state) != B:
            raise ValueError("BaselineRowEncoder needs one authoritative gravity state per window")
        n_slots = max(len(texts) for texts in sensor_texts)
        descriptors, sensor_text_ids = self.text_encoder_descriptors(sensor_texts, device, n_slots)

        # Reconstruct the contiguous native-rate signal without allowing padded patch storage to
        # leak into it. Resampling happens below in homogeneous (rate, length) groups.
        valid_len = (patch_len.clamp_min(0) * patch_padding_mask.long())                 # (B,P)
        cumulative = torch.cat(
            [valid_len.new_zeros((B, 1)), valid_len.cumsum(dim=1)], dim=1).float()       # (B,P+1)
        # ~0.1% of corpus windows (wisdm, capture24, opportunity) carry a SINGLE sample. Our
        # encoder still emits rows for them, so refusing them here would silently shrink the row
        # population and stop the arms being comparable. A one-sample window wrap-pads to a
        # constant signal, which is what the baselines' own adapters do with a short recording.
        total = cumulative[:, -1].clamp_min(1.0)                                         # (B,)
        max_total = int(total.max().item())

        def gather_samples(index: torch.Tensor) -> torch.Tensor:
            """(B,L) integer sample indices into the concatenated valid signal."""
            patch_of = torch.searchsorted(cumulative, index.contiguous(), right=True) - 1
            patch_of = patch_of.clamp(0, P - 1)
            offset = (index - cumulative.gather(1, patch_of)).long().clamp_min(0)
            offset = torch.minimum(offset, (patch_len.gather(1, patch_of) - 1).clamp_min(0))
            flat = patches.reshape(B, P * S, C)
            return flat.gather(1, (patch_of * S + offset).unsqueeze(-1).expand(-1, -1, C))

        native_index = torch.arange(max_total, device=device).view(1, -1).expand(B, -1)
        native_index = torch.minimum(native_index, (total.long() - 1).unsqueeze(1))
        window = gather_samples(native_index)                                             # (B,Tmax,C)

        # under autocast the projection returns bf16; build the row buffer in the SAME dtype
        row_dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) \
            else patches.dtype
        rows = patches.new_zeros((B, n_slots, self.d_model), dtype=row_dtype)
        present = torch.zeros((B, n_slots), dtype=torch.bool, device=device)
        rates_host = rates.detach().cpu().tolist()
        total_host = total.detach().cpu().to(torch.long).tolist()
        prepared_batches: list[torch.Tensor] = []
        destination_batches: list[torch.Tensor] = []
        joint_batches: list[torch.Tensor] = []
        for slot in range(n_slots):
            # Both released trunks are accelerometer-only. Keep a slot only when it owns a complete
            # accel triad and the stream retains gravity. Gyroscope and linear-accel rows are absent.
            live_acc = (sensor_id[:, :3] == slot) & channel_mask[:, :3]
            enough = live_acc.sum(dim=1) >= 3
            gravity_ok = torch.tensor(
                [state == "present" for state in gravity_state], dtype=torch.bool, device=device,
            )
            keep = enough & gravity_ok
            if not bool(keep.any()):
                continue
            order = torch.argsort((~live_acc).to(torch.int8), dim=1, stable=True)[:, :3]
            # Grouping preserves the official anti-aliased adapter contract without padding real
            # samples to a shared length before the sinc filter.
            kept_rows = torch.nonzero(keep, as_tuple=True)[0]
            groups: dict[tuple[float, int], list[int]] = {}
            for row in kept_rows.detach().cpu().tolist():
                groups.setdefault((float(rates_host[row]), int(total_host[row])), []).append(row)
            for (rate, length), indices in groups.items():
                group = torch.tensor(indices, dtype=torch.long, device=device)
                sel = order.index_select(0, group)
                sub = window.index_select(0, group)[:, :length]
                triad = torch.gather(sub, 2, sel.unsqueeze(1).expand(-1, length, -1)).float()
                resampled = self._resample(triad, rate)
                prepare_length = (
                    _center_crop_or_wrap if self.backbone_name == "harnet"
                    else _start_crop_or_wrap
                )
                prepared = prepare_length(resampled, self.in_len)
                prepared_batches.append(prepared)
                destination_batches.append(torch.stack(
                    [group, torch.full_like(group, slot)], dim=1,
                ))
                if self.backbone_name == "unimts":
                    joint_batches.append(torch.tensor(
                        [_stream_joint(streams[row]) for row in indices],
                        dtype=torch.long, device=device,
                    ))

        if prepared_batches:
            prepared = torch.cat(prepared_batches, dim=0)
            destination = torch.cat(destination_batches, dim=0)
            joint = torch.cat(joint_batches, dim=0) if joint_batches else None
            # Rate/length groups are a preprocessing concern, not a model-batching constraint.
            # Calling the frozen trunk once per group produced tens of thousands of tiny conv calls
            # per optimizer step. Large inference chunks retain exact prepared inputs while filling
            # the GPU. UniMTS has a much wider 22-joint activation and therefore uses a smaller cap.
            feature_batch = 4096 if self.backbone_name == "harnet" else 512
            features = []
            with torch.no_grad():
                for start in range(0, len(prepared), feature_batch):
                    stop = min(start + feature_batch, len(prepared))
                    feature_input = prepared[start:stop]
                    feature_joint = None if joint is None else joint[start:stop]
                    valid = stop - start
                    use_compiled = self.backbone_name == "unimts" and self._compiled_net is not None
                    if use_compiled and valid < feature_batch:
                        # A fixed batch avoids one graph compilation for every possible episode
                        # size. BatchNorm is frozen/eval, so repeated padding rows cannot influence
                        # real rows; their outputs are discarded immediately.
                        pad = feature_batch - valid
                        feature_input = torch.cat(
                            [feature_input, feature_input[-1:].expand(pad, -1, -1)], dim=0,
                        )
                        feature_joint = torch.cat(
                            [feature_joint, feature_joint[-1:].expand(pad)], dim=0,
                        )
                    out = self._features(feature_input, feature_joint, compiled=use_compiled)
                    # Inductor's reduce-overhead mode reuses a CUDA-graph output buffer on the
                    # next invocation. Materialize each chunk before that reuse; without the clone,
                    # multi-chunk episode batches either fail loudly or alias earlier features.
                    features.append(out[:valid].clone() if use_compiled else out[:valid])
            feature = torch.cat(features, dim=0)
            # Keep the numerical interface in FP32 even under autocast; the evidence rows may then
            # be stored in the active mixed-precision dtype.
            with torch.autocast(device_type=device.type, enabled=False):
                normalized = self.row_norm(self.proj(feature.float()))
            rows[destination[:, 0], destination[:, 1]] = normalized.to(rows.dtype)
            present[destination[:, 0], destination[:, 1]] = True

        # a slot with no description is not a row, exactly as in SetTokenizerEncoder
        present = present & sensor_text_ids.ge(0)
        dense = descriptors.index_select(0, sensor_text_ids.clamp_min(0).reshape(-1))
        dense = dense.reshape(B, n_slots, -1)
        retrieval = rows.unsqueeze(1)
        return {"retrieval_tokens": retrieval, "sensor_present": present,
                "retrieval_window_rows": True,
                "tokens": retrieval, "per_patch": None, "pooled": None,
                "sensor_context": None, "descriptor": dense, "descriptor_pred": None}

    def text_encoder_descriptors(self, sensor_texts, device, n_slots):
        """Frozen pooled sensor descriptors + the padded inverse-ID table (-1 = no sensor)."""
        flat = [text for texts in sensor_texts for text in texts]
        unique = list(dict.fromkeys(flat))
        descriptors = self.text_encoder.encode_pooled(unique, device=device)
        lookup = {text: i for i, text in enumerate(unique)}
        ids = torch.full((len(sensor_texts), n_slots), -1, dtype=torch.long)
        for b, texts in enumerate(sensor_texts):
            ids[b, :len(texts)] = torch.tensor([lookup[t] for t in texts], dtype=torch.long)
        return descriptors, ids.to(device)
