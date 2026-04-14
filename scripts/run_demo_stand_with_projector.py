#!/usr/bin/env python3
r"""
Полный цикл демо-стенда с проектором:

1. Обучение DeepMahalanobisProjector (triplet loss) на CSV.
2. Обучение MahalanobisDriftDetector с этим проектором.
3. Сохранение ``models/demo_stand/demo_model.joblib`` и парных файлов проектора.

Далее из корня репозитория::

    streamlit run src/mcd/app_streamlit.py

В боковой панели выберите «Load from disk» и путь
``models/demo_stand/demo_model.joblib``.

Запуск::

    pip install -e ".[train]"
    python scripts/run_demo_stand_with_projector.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

# Корень репозитория (родитель каталога scripts/)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "demo_labeled.csv"
MODEL_DIR = ROOT / "models" / "demo_stand"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV с колонками subject, body, label")
    parser.add_argument("--label-column", default="queue")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--skip-projector-train", action="store_true", help="Использовать уже обученные веса в demo_stand")
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=1.0,
        help="Квантиль по внутрикластерным расстояниям; 1.0 = max (разумно при n≪d в демо).",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0,
        help="Диагональная регуляризация ковариации (1e-6 мало при малом n и dim=32 после проектора).",
    )
    parser.add_argument("--min-cluster-size", type=int, default=3)
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_joblib = MODEL_DIR / "demo_model.joblib"
    # Имена должны совпадать с тем, что ожидает MahalanobisDriftDetector.save(load) для stem demo_model
    proj_pt = MODEL_DIR / "demo_model_projector.pt"
    proj_cfg = MODEL_DIR / "demo_model_projector_config.json"

    if not args.skip_projector_train:
        train_py = ROOT / "scripts" / "train_projector.py"
        cmd = [
            sys.executable,
            str(train_py),
            "--csv",
            str(args.csv),
            "--label-column",
            args.label_column,
            "--output",
            str(proj_pt),
            "--config-out",
            str(proj_cfg),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            "8",
            "--embed-batch-size",
            "16",
            "--output-dim",
            "32",
            "--hidden-dims",
            "64,32",
            "--lr",
            "0.001",
        ]
        print("Step 1/2: training projector...\n ", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(ROOT))
    elif not proj_pt.is_file() or not proj_cfg.is_file():
        raise SystemExit(f"Missing {proj_pt} or {proj_cfg}; run without --skip-projector-train")

    from src.mcd.io import load_labeled_tickets_csv
    from src.mcd.modeling.classifier import MahalanobisDriftDetector
    from src.mcd.projection.projector import DeepMahalanobisProjector

    print("Step 2/2: fitting MahalanobisDriftDetector + save...")
    with open(proj_cfg, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("schema", None)
    projector = DeepMahalanobisProjector.from_architecture_dict(cfg)
    load_kw: dict[str, object] = {"map_location": torch.device("cpu")}
    try:
        state = torch.load(proj_pt, weights_only=True, **load_kw)
    except TypeError:
        state = torch.load(proj_pt, **load_kw)
    projector.load_state_dict(state)
    projector.eval()

    texts, labels, _, _ = load_labeled_tickets_csv(str(args.csv), args.label_column)
    detector = MahalanobisDriftDetector(
        threshold_quantile=args.threshold_quantile,
        min_cluster_size=args.min_cluster_size,
        projector=projector,
        regularization=args.regularization,
    )
    detector.fit(texts, labels)
    detector.save(str(model_joblib))

    print("\nDone.")
    print("  Model:", model_joblib.resolve())
    print("  Projector:", proj_pt.resolve())
    print("\nStart UI from repo root:")
    print("  streamlit run src/mcd/app_streamlit.py")
    print('Sidebar: "Load from disk" and path:')
    print(f"  {model_joblib.as_posix()}")


if __name__ == "__main__":
    main()
