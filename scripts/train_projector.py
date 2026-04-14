#!/usr/bin/env python3
"""Train DeepMahalanobisProjector on labeled ticket embeddings (triplet loss)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.mcd.embedding.sbert import SBERT
from src.mcd.io import load_labeled_tickets_csv
from src.mcd.projection.projector import DeepMahalanobisProjector

logger = logging.getLogger(__name__)


def parse_hidden_dims(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def embed_texts_batched(embedder: SBERT, texts: list[str], batch_size: int) -> NDArray[np.float64]:
    chunks: list[NDArray[np.float64]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        chunks.append(embedder.embed(batch))
    stacked = np.vstack(chunks)
    return stacked.astype(np.float64, copy=False)


def triplet_loss_step(
    model: nn.Module,
    criterion: nn.TripletMarginLoss,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    """One batch: random valid triplets (anchor, positive, negative)."""
    device = embeddings.device
    b = embeddings.size(0)
    anc_list: list[torch.Tensor] = []
    pos_list: list[torch.Tensor] = []
    neg_list: list[torch.Tensor] = []

    for i in range(b):
        li = labels[i]
        same_idx = torch.nonzero(labels == li, as_tuple=False).squeeze(-1)
        diff_idx = torch.nonzero(labels != li, as_tuple=False).squeeze(-1)
        same_others = same_idx[same_idx != i]
        if same_others.numel() == 0 or diff_idx.numel() == 0:
            continue
        ji = int(same_others[torch.randint(same_others.numel(), (1,), device=device)].item())
        ki = int(diff_idx[torch.randint(diff_idx.numel(), (1,), device=device)].item())
        anc_list.append(embeddings[i])
        pos_list.append(embeddings[ji])
        neg_list.append(embeddings[ki])

    if not anc_list:
        return None
    return cast(
        torch.Tensor,
        criterion(
            torch.stack(anc_list),
            torch.stack(pos_list),
            torch.stack(neg_list),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Path to labeled CSV (subject, body, label column)")
    parser.add_argument("--label-column", required=True, help="Name of the label / queue column")
    parser.add_argument("--output", type=Path, default=Path("projector.pt"), help="Path for state_dict (.pt)")
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Path for architecture JSON (default: same stem as --output + _config.json)",
    )
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Batch size for SBERT.encode")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=str, default="256,128", help="Comma-separated hidden layer sizes")
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=1.0, help="Triplet margin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    texts, labels, _, _ = load_labeled_tickets_csv(str(args.csv), args.label_column)
    unique = sorted(set(labels))
    if len(unique) < 2:
        logger.error("Need at least two distinct labels for triplet training, got %s", len(unique))
        sys.exit(1)

    label_to_idx = {lb: i for i, lb in enumerate(unique)}
    y = np.array([label_to_idx[lb] for lb in labels], dtype=np.int64)

    logger.info("Embedding %s texts with %s", len(texts), args.sbert_model)
    embedder = SBERT(model_name=args.sbert_model)
    x_np = embed_texts_batched(embedder, texts, args.embed_batch_size)
    input_dim = x_np.shape[1]

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    dataset = TensorDataset(x, y_t)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model = DeepMahalanobisProjector(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        n_batches = 0
        for xb, yb in loader:
            loss_t = triplet_loss_step(model, criterion, xb, yb)
            if loss_t is None:
                continue
            optimizer.zero_grad()
            loss_t.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss_t.item())
            n_batches += 1
        if n_batches:
            logger.info("epoch %s/%s loss=%.6f", epoch + 1, args.epochs, total / n_batches)
        else:
            logger.warning("epoch %s: no valid triplets in any batch", epoch + 1)

    model.eval()
    cfg_path = args.config_out
    if cfg_path is None:
        cfg_path = args.output.with_name(args.output.stem + "_config.json")

    arch = model.architecture_dict()
    arch["schema"] = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(arch, f, indent=2)
    torch.save(model.cpu().state_dict(), args.output)
    logger.info("Saved weights to %s and config to %s", args.output.resolve(), cfg_path.resolve())


if __name__ == "__main__":
    main()
