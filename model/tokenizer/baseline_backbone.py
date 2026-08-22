"""Pretrained third-party backbones wearing our encoder's output contract.

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
   Running either on a 1 s slice is far outside its training distribution, so each backbone sees
   the WHOLE 6 s window once per sensor and its embedding is broadcast across that window's patch
   positions. Rows within a window therefore become identical -- a real reduction in bank
   diversity, and the honest cost of using a window-level encoder in a patch-level engine.

2. NON-ACCEL SENSORS. Both backbones are accelerometer-only. A gyroscope slot is still encoded
   (its three channels are fed in the accelerometer's place) rather than dropped, so the row
   POPULATION matches our encoder's exactly and the comparison isolates representation quality
   rather than which rows exist. Out-of-distribution for the backbone; disclosed, not hidden.

The projection to the engine's width is always trainable -- a frozen backbone whose output could
not be linearly re-based would be testing the projection, not the representation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _TrunkShim(nn.Module):
    """Publishes the layer count the engine reads; the backbone has no trunk of ours."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.num_layers = int(n_layers)


def _resample_to(x: torch.Tensor, length: int) -> torch.Tensor:
    """(M, T, C) -> (M, length, C) by linear interpolation along time (differentiable)."""
    if x.shape[1] == length:
        return x
    return torch.nn.functional.interpolate(
        x.permute(0, 2, 1), size=length, mode="linear", align_corners=True
    ).permute(0, 2, 1)


class BaselineRowEncoder(nn.Module):
    """A pretrained backbone presented to the engine as a sensor-granularity row encoder."""

    #: guards in `encode_batch` check these; the stand-in satisfies the same contract
    trunk = "temporal"
    token_granularity = "sensor"
    use_sensor_isolated_retrieval = True

    def __init__(self, backbone: str, d_model: int = 128, freeze: bool = True,
                 device: torch.device | None = None):
        super().__init__()
        if backbone not in {"harnet", "unimts"}:
            raise ValueError("backbone must be 'harnet' or 'unimts'")
        self.backbone_name = backbone
        self.freeze = bool(freeze)
        device = device or torch.device("cpu")

        if backbone == "harnet":
            from baselines.harnet.adapter import _load_harnet
            module = _load_harnet(2, device)
            # harnet's feature extractor is the ResNet trunk with its classifier removed
            self.net = getattr(module, "feature_extractor", module)
            self.in_hz, self.in_len, self.out_dim = 30.0, 150, 512
        else:
            from baselines.unimts.adapter import UNIMTS_CKPT, UNIMTS_REPO, N_JOINTS
            import sys
            if str(UNIMTS_REPO) not in sys.path:
                sys.path.insert(0, str(UNIMTS_REPO))
            from baselines.unimts.adapter import UniMTSAdapter
            self._uni = UniMTSAdapter()
            self.net = self._uni.setup(device)["model"]
            self.n_joints = N_JOINTS
            self.in_hz, self.in_len, self.out_dim = 20.0, 200, 512

        # Both loaders hand back a model with gradients already disabled (they exist to serve
        # frozen-feature baselines), so the fine-tuned arm must switch them back ON explicitly --
        # otherwise "fine-tuned" silently trains the projection only.
        self.net.train(not self.freeze)
        for name, parameter in self.net.named_parameters():
            # UniMTS's checkpoint carries its whole CLIP text tower (63M of its 68M parameters).
            # We read text through the same frozen MiniLM every other arm uses, so only the
            # accelerometer ST-GCN (`model.acc.*`, 5.2M) is the encoder under test.
            trainable = (not self.freeze) and (
                backbone != "unimts" or name.startswith("model.acc.")
            )
            parameter.requires_grad_(trainable)
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
    def _features(self, window: torch.Tensor, joint: torch.Tensor | None) -> torch.Tensor:
        """(M, T, 3) native-unit accel -> (M, out_dim)."""
        x = window if window.shape[1] == self.in_len else _resample_to(window, self.in_len)
        if self.backbone_name == "harnet":
            return self.net(x.permute(0, 2, 1)).flatten(1)          # (M, 3, 150) -> (M, 1024)
        # UniMTS: place the 3 channels at one SMPL joint, zeros elsewhere -> (M,3,200,22,1)
        M, T, _ = x.shape
        grid = x.new_zeros((M, T, self.n_joints, 3))
        idx = joint if joint is not None else torch.zeros(M, dtype=torch.long, device=x.device)
        grid[torch.arange(M, device=x.device), :, idx, :] = x * 9.80665   # g -> m/s^2
        return self.net.encode_image(grid.permute(0, 3, 1, 2).unsqueeze(-1))

    # ---------------------------------------------------------------- the encoder contract
    def forward(self, patches, rates, patch_len, role_texts, positions, *,
                patch_durations=None, channel_mask=None, patch_padding_mask=None,
                sensor_texts=None, sensor_id=None, source_rate_hz=None,
                return_retrieval_tokens=True, joint_ids=None, **_ignored):
        B, P, S, C = patches.shape
        device = patches.device
        if sensor_id is None or channel_mask is None or patch_padding_mask is None:
            raise ValueError("BaselineRowEncoder needs sensor_id, channel_mask, patch_padding_mask")
        if not sensor_texts:
            raise ValueError("BaselineRowEncoder needs sensor_texts (one description per sensor)")
        n_slots = max(len(texts) for texts in sensor_texts)
        descriptors, sensor_text_ids = self.text_encoder_descriptors(sensor_texts, device, n_slots)

        # --- rebuild each item's contiguous signal at the BACKBONE's sampling rate -------------
        # Padding slots and heterogeneous rates make the naive `reshape(B, P*S, C)` wrong twice
        # over: it interleaves a 20 Hz item's 20 real samples with 80 zeros when the batch also
        # holds a 100 Hz item, and it resamples by length rather than by rate, time-compressing a
        # 6 s window into harnet's 5 s receptive field. Both would handicap the backbone with an
        # artefact and be invisible in the loss. Instead: index the concatenated VALID samples,
        # resample at the item's own rate, and wrap-pad to the backbone's window -- the same
        # contract each baseline's own adapter implements.
        valid_len = (patch_len.clamp_min(0) * patch_padding_mask.long())                 # (B,P)
        cumulative = torch.cat(
            [valid_len.new_zeros((B, 1)), valid_len.cumsum(dim=1)], dim=1).float()       # (B,P+1)
        # ~0.1% of corpus windows (wisdm, capture24, opportunity) carry a SINGLE sample. Our
        # encoder still emits rows for them, so refusing them here would silently shrink the row
        # population and stop the arms being comparable. A one-sample window wrap-pads to a
        # constant signal, which is what the baselines' own adapters do with a short recording.
        total = cumulative[:, -1].clamp_min(1.0)                                         # (B,)
        duration_s = total / rates.clamp_min(1e-6)
        target_n = (duration_s * self.in_hz).round().clamp_min(1.0)                      # (B,)
        step = torch.arange(self.in_len, device=device, dtype=torch.float32).view(1, -1)
        # wrap-pad when the window is shorter than the backbone's input, truncate when longer
        wrapped = torch.remainder(step, target_n.unsqueeze(1))
        position = wrapped * rates.unsqueeze(1) / self.in_hz
        position = torch.minimum(position, (total - 1).unsqueeze(1))                     # (B,L)
        low = position.floor()
        high = torch.minimum(low + 1, (total - 1).unsqueeze(1))
        frac = (position - low).unsqueeze(-1)

        def gather_samples(index: torch.Tensor) -> torch.Tensor:
            """(B,L) sample indices into the concatenated valid signal -> (B,L,C)."""
            patch_of = torch.searchsorted(cumulative, index.contiguous(), right=True) - 1
            patch_of = patch_of.clamp(0, P - 1)
            offset = (index - cumulative.gather(1, patch_of)).long().clamp_min(0)
            offset = torch.minimum(offset, (patch_len.gather(1, patch_of) - 1).clamp_min(0))
            flat = patches.reshape(B, P * S, C)
            return flat.gather(1, (patch_of * S + offset).unsqueeze(-1).expand(-1, -1, C))

        window = gather_samples(low) * (1.0 - frac) + gather_samples(high) * frac        # (B,L,C)

        # under autocast the projection returns bf16; build the row buffer in the SAME dtype
        row_dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) \
            else patches.dtype
        rows = patches.new_zeros((B, n_slots, self.d_model), dtype=row_dtype)
        present = torch.zeros((B, n_slots), dtype=torch.bool, device=device)
        for slot in range(n_slots):
            live = (sensor_id == slot) & channel_mask                            # (B, C)
            has = live.any(dim=1)
            if not bool(has.any()):
                continue
            # take this slot's first three live channels (accel triad, or gyro triad)
            order = torch.argsort((~live).to(torch.int8), dim=1, stable=True)[:, :3]   # (B,3)
            enough = live.sum(dim=1) >= 3
            keep = has & enough
            if not bool(keep.any()):
                continue
            sel = order[keep]                                                    # (M,3)
            sub = window[keep]                                                   # (M,in_len,C)
            triad = torch.gather(sub, 2, sel.unsqueeze(1).expand(-1, sub.shape[1], -1))
            joint = None if joint_ids is None else joint_ids.to(device)[keep]
            with torch.set_grad_enabled(self.training and not self.freeze):
                feature = self._features(triad.float(), joint)
            rows[keep, slot] = self.row_norm(self.proj(feature.to(rows.dtype)))
            present[keep, slot] = True

        # a slot with no description is not a row, exactly as in SetTokenizerEncoder
        present = present & sensor_text_ids.ge(0)
        dense = descriptors.index_select(0, sensor_text_ids.clamp_min(0).reshape(-1))
        dense = dense.reshape(B, n_slots, -1)
        retrieval = rows.unsqueeze(1).expand(B, P, n_slots, self.d_model).contiguous()
        return {"retrieval_tokens": retrieval, "sensor_present": present,
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
