"""Self-supervised LiMU-BERT pretraining ON OUR TRAINING CORPUS.

LiMU-BERT ships no released weights; it is an SSL *method* (masked
reconstruction). This script pools the 9 non-eval training datasets' grids into
LiMU-BERT's input contract (6-ch acc+gyro, 20 Hz, 120 samples — see
:mod:`baselines.limubert.prep`) and drives the UPSTREAM LiMU-BERT SSL pipeline
(model + Trainer + masking + ÷9.8 normalization, reused from
``auxiliary_repos/LIMU-BERT-Public`` via ``sys.path``) to produce the backbone
checkpoint ``baselines/limubert/limubert_backbone.pt`` that the adapter loads.

  * Vendoring the upstream repo under ``baselines/limubert/repo/`` is a follow-up;
    for now it is imported from the legacy ``auxiliary_repos`` tree.
  * Run length is a CLI arg so a SMOKE run (few epochs, small subset) proves the
    pipeline end-to-end while the FULL run reproduces the paper-scale pretrain.

FULL (paper-scale) pretrain — the deferred compute job:

    python -m baselines.limubert.train --epochs 3200 --batch-size 128

SMOKE (what we run to prove the wiring; under-trained by design):

    python -m baselines.limubert.train --epochs 2 --batch-size 128 \
        --max-per-stream 800
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.limubert import prep

# Upstream LiMU-BERT repo (reused via sys.path; vendoring under repo/ is a follow-up).
LIMU_REPO = Path("/home/alex/code/HALO/legacy_code/auxiliary_repos/LIMU-BERT-Public")

_HERE = Path(__file__).resolve().parent
BACKBONE_CKPT = _HERE / "limubert_backbone.pt"


def _seed_worker(worker_id):
    """Reseed numpy AND stdlib-random per DataLoader worker.

    LiMU-BERT's span-mask draws from both ``np.random`` (utils.span_mask) and
    stdlib ``random`` (utils.bert_mask -> random.sample). Forked workers inherit
    identical global RNG state, so without this they emit correlated/duplicate
    masks across workers -- shrinking mask diversity and changing the SSL
    objective. torch assigns each worker a unique ``initial_seed()``; derive both
    RNGs from it so masking stays diverse AND the run stays reproducible.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Self-pretrain LiMU-BERT on our corpus")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=3431)
    ap.add_argument("--max-per-stream", type=int, default=800,
                    help="per-stream window cap. DEFAULT 800 = smoke; omitting KEEPS this cap. "
                         "Pass 0 for the full corpus, or e.g. 20000 for the balanced corpus.")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args(argv)

    if str(LIMU_REPO) not in sys.path:
        sys.path.insert(0, str(LIMU_REPO))
    import models as lb_models
    import train as lb_train
    from config import PretrainModelConfig, TrainConfig, MaskConfig
    from utils import (LIBERTDataset4Pretrain, Preprocess4Mask, Preprocess4Normalization,
                       prepare_pretrain_dataset, get_device)

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print(f"[limubert] device={device}")

    cap = args.max_per_stream or None
    data = prep.build_pretrain_array(max_per_stream=cap, seed=args.seed)
    labels = np.zeros((data.shape[0], data.shape[1], 2), dtype=np.float32)  # dummy (SSL)
    print(f"[limubert] corpus: {data.shape[0]} windows, shape {data.shape}")

    # Apply LiMU-BERT's deterministic accel normalization ONCE to the pooled array
    # instead of inside every __getitem__. For 6-ch data Preprocess4Normalization is
    # exactly accel[:3] /= 9.8 (the magnetometer branch only fires at 9 channels), so
    # this is value-identical and lets the per-item pipeline drop to masking-only.
    data[:, :, :3] /= 9.8

    model_cfg = PretrainModelConfig(hidden=72, hidden_ff=144, feature_num=6,
                                    n_layers=4, n_heads=4, seq_len=120, emb_norm=True)
    train_cfg = TrainConfig(seed=args.seed, batch_size=args.batch_size, lr=args.lr,
                            n_epochs=args.epochs, warmup=0.1, save_steps=100000,
                            total_steps=200000000)
    mask_cfg = MaskConfig(mask_ratio=0.15, mask_alpha=6, max_gram=10,
                          mask_prob=0.8, replace_prob=0.0)

    # Normalization applied once above (value-identical); per-item work is now masking only.
    pipeline = [Preprocess4Mask(mask_cfg)]
    d_train, _, d_test, _ = prepare_pretrain_dataset(data, labels, 0.8, seed=train_cfg.seed)
    print(f"[limubert] train={len(d_train)} val={len(d_test)}")

    ds_train = LIBERTDataset4Pretrain(d_train, pipeline=pipeline)
    ds_test = LIBERTDataset4Pretrain(d_test, pipeline=pipeline)
    # num_workers>0 overlaps the CPU span-masking (the measured bottleneck) with GPU
    # compute; _seed_worker keeps each worker's mask RNG independent (see above).
    # pin_memory + persistent_workers + prefetch trim per-epoch overhead. Pure-speed,
    # faithful (identical objective, just parallelized). Measured ~1.6x on the box.
    ld_train = DataLoader(ds_train, shuffle=True, batch_size=train_cfg.batch_size,
                          num_workers=4, pin_memory=True, persistent_workers=True,
                          prefetch_factor=4, worker_init_fn=_seed_worker)
    ld_test = DataLoader(ds_test, shuffle=False, batch_size=train_cfg.batch_size,
                         num_workers=2, pin_memory=True, persistent_workers=True,
                         prefetch_factor=4, worker_init_fn=_seed_worker)

    model = lb_models.LIMUBertModel4Pretrain(model_cfg)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    save_prefix = _HERE / "pretrain_run" / "model"
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    trainer = lb_train.Trainer(train_cfg, model, optimizer, str(save_prefix), device)

    def func_loss(m, batch):
        mask_seqs, masked_pos, seqs = batch
        return criterion(m(mask_seqs, masked_pos), seqs)

    def func_forward(m, batch):
        mask_seqs, masked_pos, seqs = batch
        return m(mask_seqs, masked_pos), seqs

    def func_evaluate(seqs, pred):
        return criterion(pred, seqs).mean().cpu().numpy()

    trainer.pretrain(func_loss, func_forward, func_evaluate, ld_train, ld_test)

    # Trainer.pretrain reloads the best state into `model`; save it as the
    # canonical checkpoint the adapter loads.
    torch.save(model.state_dict(), str(BACKBONE_CKPT))
    print(f"[limubert] saved backbone -> {BACKBONE_CKPT}")


if __name__ == "__main__":
    main()
