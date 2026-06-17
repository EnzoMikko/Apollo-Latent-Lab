"""Sweep de la profondeur latente k + figures + rapport MVP.

Lance latent(k) pour k dans une liste donnée, puis une baseline compute-matched
optionnelle, collecte les métriques, produit les figures et remplit le rapport.

Exemple :
    python scripts/run_sweep.py --base-config configs/latent_addition.yaml \
        --k 0 1 2 4 8 --epochs 15
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neural_latent_lab.training.train import train  # noqa: E402
from neural_latent_lab.utils.config import load_config  # noqa: E402


def _load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["metrics"]


def run_sweep(base_config: str, ks: list[int], epochs: int | None, seed: int) -> list[dict]:
    results = []
    for k in ks:
        cfg = load_config(base_config)
        cfg.seed = seed
        cfg.model.kind = "latent"
        cfg.model.k = k
        cfg.name = f"latent_addition_k{k}"
        if epochs is not None:
            cfg.train.epochs = epochs
        path = train(cfg)
        results.append(_load_metrics(path))
    return results


def make_figures(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [r["latent_depth_k"] for r in results]

    def plot(ys, ylabel, fname, title):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, ys, marker="o")
        ax.set_xlabel("k (cycles latents)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)

    plot([r["test_accuracy"] for r in results], "test accuracy", "accuracy_vs_k.png", "Accuracy vs k")
    plot([r["ood_accuracy"] for r in results], "OOD accuracy", "ood_accuracy_vs_k.png", "OOD accuracy vs k")
    plot([r["accuracy_per_flop"] for r in results], "accuracy / FLOP", "accuracy_per_flop_vs_k.png", "Accuracy per FLOP vs k")
    plot([r["latency_ms"] for r in results], "latence (ms/ex)", "latency_vs_k.png", "Latence vs k")


def write_report(results: list[dict], template: Path, out: Path) -> None:
    header = "| k | params | test_acc | ood_acc | flops | acc/flop | latency_ms |"
    sep = "|---|---|---|---|---|---|---|"
    rows = [
        f"| {r['latent_depth_k']} | {r['num_parameters']:,} | {r['test_accuracy']:.3f} | "
        f"{r['ood_accuracy']:.3f} | {r['estimated_flops']:.2e} | "
        f"{r['accuracy_per_flop']:.2e} | {r['latency_ms']:.3f} |"
        for r in results
    ]
    table = "\n".join([header, sep] + rows)
    text = template.read_text(encoding="utf-8")
    text = text.replace("<!-- SWEEP_TABLE -->", table)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Rapport écrit : {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/latent_addition.yaml")
    parser.add_argument("--k", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_sweep(args.base_config, args.k, args.epochs, args.seed)
    out_dir = Path("outputs")
    make_figures(results, out_dir / "figures")
    write_report(
        results,
        REPO / "reports" / "mvp_report.template.md",
        REPO / "reports" / "mvp_report.md",
    )
    # Sauvegarde brute du sweep pour ré-analyse.
    with open(out_dir / "metrics" / "sweep_summary.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
