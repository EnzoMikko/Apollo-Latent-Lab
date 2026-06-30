"""Point d'entrée en ligne de commande.

Exemples :
    python -m neural_latent_lab.cli train --config configs/baseline_addition.yaml
    python -m neural_latent_lab.cli analyze --checkpoint outputs/checkpoints/<id>.pt
"""

from __future__ import annotations

import argparse

from neural_latent_lab.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="nll", description="Neural Latent Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Entraîner un modèle depuis une config")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--seed", type=int, default=None, help="Override la seed de la config")

    p_an = sub.add_parser("analyze", help="Extraire et analyser les latents d'un checkpoint")
    p_an.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    if args.command == "train":
        from neural_latent_lab.training.train import train

        cfg = load_config(args.config)
        if args.seed is not None:
            cfg.seed = args.seed
        train(cfg)
    elif args.command == "analyze":
        from neural_latent_lab.analysis.latents import analyze_checkpoint

        analyze_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
