"""Self-supervised CrossHAR pretraining ON OUR TRAINING CORPUS.

CrossHAR ships no released weights; it is an SSL *method* (masked reconstruction +
contrastive NT-Xent). This script pools the 9 non-eval training datasets' grids
into CrossHAR's input contract (6-ch acc+gyro, 20 Hz, 120 samples — see
:mod:`baselines.crosshar.prep`) and drives the UPSTREAM CrossHAR SSL pipeline
(model + Trainer + Contrastive + NT-Xent loss + masking/augmentation, reused from
``auxiliary_repos/CrossHAR`` via ``sys.path``) to produce the backbone checkpoint
``baselines/crosshar/crosshar_backbone.pt`` that the adapter loads.

  * Vendoring the upstream repo under ``baselines/crosshar/repo/`` is a follow-up;
    for now it is imported from the legacy ``auxiliary_repos`` tree.
  * Run length is a CLI arg so a SMOKE run (few epochs, small subset, no channel
    augmentation) proves the pipeline end-to-end, while the FULL run reproduces
    the paper-scale pretrain.

FULL (paper-scale) pretrain — the deferred compute job:

    python -m baselines.crosshar.train --epochs 1600 --epochs-cl 800 \
        --batch-size 512 --augment

SMOKE (what we run to prove the wiring; under-trained by design):

    python -m baselines.crosshar.train --epochs 2 --epochs-cl 1 \
        --batch-size 256 --max-per-stream 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.crosshar import prep

# Upstream CrossHAR repo (reused via sys.path; vendoring under repo/ is a follow-up).
CROSSHAR_REPO = Path("/home/alex/code/HALO/legacy_code/auxiliary_repos/CrossHAR")

_HERE = Path(__file__).resolve().parent
BACKBONE_CKPT = _HERE / "crosshar_backbone.pt"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Self-pretrain CrossHAR on our corpus")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--epochs-cl", type=int, default=1,
                    help="# of final epochs that add the contrastive loss")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=3431)
    ap.add_argument("--max-per-stream", type=int, default=800,
                    help="per-stream window cap. DEFAULT 800 = smoke; omitting KEEPS this cap. "
                         "Pass 0 for the full corpus, or e.g. 20000 for the balanced corpus.")
    ap.add_argument("--augment", action="store_true",
                    help="apply upstream channel_aug (6x); off for smoke speed")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args(argv)

    if str(CROSSHAR_REPO) not in sys.path:
        sys.path.insert(0, str(CROSSHAR_REPO))
    import models as ch_models
    import train as ch_train
    from Contrastive import Contrastive
    from config import PretrainModelConfig, TrainConfig, MaskConfig
    from utils import (Dataset4Pretrain, Preprocess4Mask, prepare_pretrain_dataset,
                       augument_dataset)

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"[crosshar] device={device}")

    cap = args.max_per_stream or None
    data = prep.build_pretrain_array(max_per_stream=cap, seed=args.seed)
    labels = np.zeros((data.shape[0], data.shape[1], 2), dtype=np.float32)  # dummy (SSL)
    print(f"[crosshar] corpus: {data.shape[0]} windows, shape {data.shape}")

    if args.augment:
        data, labels = augument_dataset(data, labels, method="channel_aug")
        print(f"[crosshar] after channel_aug: {data.shape[0]} windows")

    model_cfg = PretrainModelConfig(feature_num=6, hidden=72, hidden_ff=144,
                                    n_layers=1, n_heads=4, seq_len=120, emb_norm=True)
    train_cfg = TrainConfig(seed=args.seed, batch_size=args.batch_size, lr=args.lr,
                            n_epochs=args.epochs, n_epochs_cl=args.epochs_cl,
                            warmup=0.1, save_steps=100000, total_steps=200000000)
    mask_cfg = MaskConfig(mask_ratio=0.15, mask_alpha=6, max_gram=10,
                          mask_prob=0.8, replace_prob=0.0)

    pipeline = [Preprocess4Mask(mask_cfg)]
    d_train, _, d_test, _ = prepare_pretrain_dataset(data, labels, 0.8, seed=train_cfg.seed)
    print(f"[crosshar] train={len(d_train)} val={len(d_test)}")

    ds_train = Dataset4Pretrain(d_train, pipeline=pipeline)
    ds_test = Dataset4Pretrain(d_test, pipeline=pipeline)
    ld_train = DataLoader(ds_train, shuffle=True, batch_size=train_cfg.batch_size, drop_last=True)
    ld_test = DataLoader(ds_test, shuffle=False, batch_size=train_cfg.batch_size, drop_last=True)
    if len(ld_train) == 0 or len(ld_test) == 0:
        raise SystemExit(
            f"batch_size={train_cfg.batch_size} too large for train={len(d_train)}/"
            f"val={len(d_test)} windows (drop_last empties a loader); lower --batch-size "
            "or raise --max-per-stream.")

    masked_model = ch_models.MaskedModel4Pretrain(model_cfg).to(device)
    contrastive_model = Contrastive().to(device)
    criterion = nn.MSELoss(reduction="none")
    opt_m = torch.optim.Adam(masked_model.parameters(), lr=train_cfg.lr)
    opt_c = torch.optim.Adam(contrastive_model.parameters(), lr=train_cfg.lr)

    save_prefix = _HERE / "pretrain_run" / "model"
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    trainer = ch_train.Trainer(train_cfg, masked_model, opt_m, contrastive_model, opt_c,
                               str(save_prefix), device, batch_size=train_cfg.batch_size,
                               criterion=criterion)
    trainer.pretrain(ld_train, ld_test)

    # Trainer saves the best masked model to "<prefix>_masked_6_1.pt"; promote it
    # to the canonical checkpoint the adapter loads.
    best = save_prefix.parent / "model_masked_6_1.pt"
    sd = torch.load(str(best), map_location="cpu", weights_only=True)
    torch.save(sd, str(BACKBONE_CKPT))
    print(f"[crosshar] saved backbone -> {BACKBONE_CKPT}")


if __name__ == "__main__":
    main()
