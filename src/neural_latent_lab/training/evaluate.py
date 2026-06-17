"""Évaluation : perte token-level et exact-match par décodage autorégressif.

L'exact-match est la métrique de vérité : on décode la réponse en greedy depuis
le prompt et on la compare à la réponse attendue (chaîne, éventuellement inversée).
Le décodage est groupé par longueur de prompt pour être batché efficacement sur CPU.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neural_latent_lab.data.addition import AdditionDataset, AdditionVocab


def _logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    return out[0] if isinstance(out, tuple) else out


def compute_loss(model, batch, device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["loss_mask"].to(device)
    logits = _logits(model, input_ids)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none"
    ).view(labels.shape)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        total += compute_loss(model, batch, device).item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _greedy_decode_batch(model, prompts: torch.Tensor, max_new: int, eos_id: int):
    """Décode en greedy un batch de prompts (mêmes longueurs). Retourne les ids
    générés (hors prompt) par séquence, tronqués à EOS."""
    device = prompts.device
    g = prompts.size(0)
    seq = prompts
    finished = torch.zeros(g, dtype=torch.bool, device=device)
    generated = [[] for _ in range(g)]
    for _ in range(max_new):
        logits = _logits(model, seq)
        next_tok = logits[:, -1, :].argmax(dim=-1)
        for i in range(g):
            if not finished[i]:
                tok = int(next_tok[i])
                if tok == eos_id:
                    finished[i] = True
                else:
                    generated[i].append(tok)
        seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
        if finished.all():
            break
    return generated


@torch.no_grad()
def exact_match_accuracy(
    model, dataset: AdditionDataset, vocab: AdditionVocab, device, batch_size: int = 256
):
    """Accuracy exacte (chaîne réponse) + accuracy par longueur + tokens générés moyens."""
    model.eval()
    # Regroupe les indices par longueur de prompt pour batcher sans padding.
    by_len: dict[int, list[int]] = defaultdict(list)
    for idx, ex in enumerate(dataset.examples):
        by_len[len(ex.prompt)].append(idx)

    correct, total, tokens = 0, 0, 0
    per_len_correct: dict[int, int] = defaultdict(int)
    per_len_total: dict[int, int] = defaultdict(int)
    max_answer = dataset.max_len  # borne haute sûre

    for plen, indices in by_len.items():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            prompts = torch.stack(
                [torch.tensor(vocab.encode(dataset.examples[i].prompt)) for i in chunk]
            ).to(device)
            gen = _greedy_decode_batch(model, prompts, max_answer, vocab.eos_id)
            for local, i in enumerate(chunk):
                ex = dataset.examples[i]
                pred = vocab.decode(gen[local])
                ok = pred == ex.answer
                correct += int(ok)
                total += 1
                tokens += len(gen[local]) + 1  # +EOS
                per_len_correct[ex.n_digits] += int(ok)
                per_len_total[ex.n_digits] += 1

    accuracy = correct / max(total, 1)
    by_length = {
        str(d): per_len_correct[d] / per_len_total[d] for d in sorted(per_len_total)
    }
    mean_tokens = tokens / max(total, 1)
    return accuracy, by_length, mean_tokens
