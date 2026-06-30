"""Dataset d'addition multi-chiffres avec test de généralisation OOD en longueur.

Conception scientifique (cf. README §5, et critiques du plan) :

* Tokenisation **char-level** : chiffres ``0-9``, ``+``, ``=``, plus PAD/EOS.
* Format **autorégressif** : la séquence complète ``"12+45=57<EOS>"`` est apprise,
  mais la perte n'est calculée que sur les tokens de réponse (après ``=``).
* ``reverse_output`` (défaut True) écrit le résultat **LSB en premier** (``"57" -> "75"``).
  C'est le trick standard de la littérature : il aligne l'ordre de génération sur
  l'ordre de propagation de la retenue, ce qui rend l'addition apprenable et la
  généralisation en longueur possible.
* Le split **OOD** n'utilise QUE des longueurs d'opérandes strictement plus grandes
  que celles du train : c'est le test de généralisation de H1.

Chaque exemple porte des **métadonnées** (longueur, présence de retenue, valeur du
résultat, succès) exploitées par ``analysis/latents.py`` pour les probes linéaires.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"


class AdditionVocab:
    """Vocabulaire char-level partagé par tous les splits."""

    def __init__(self) -> None:
        symbols = [PAD_TOKEN, EOS_TOKEN] + [str(d) for d in range(10)] + ["+", "="]
        self.stoi = {s: i for i, s in enumerate(symbols)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            s = self.itos[int(i)]
            if s == EOS_TOKEN:
                break
            if s == PAD_TOKEN:
                continue
            out.append(s)
        return "".join(out)


def _has_carry(a: int, b: int) -> bool:
    """Vrai si l'addition ``a + b`` génère au moins une retenue."""
    carry = 0
    while a > 0 or b > 0:
        s = (a % 10) + (b % 10) + carry
        carry = 1 if s >= 10 else 0
        if carry:
            return True
        a //= 10
        b //= 10
    return False


@dataclass
class AdditionExample:
    a: int
    b: int
    prompt: str  # "12+45="
    answer: str  # "57" ou "75" si reverse_output
    n_digits: int
    has_carry: bool


def _sample_operand(rng: random.Random, lo: int, hi: int) -> int:
    """Tire un entier dont le nombre de chiffres est uniforme dans [lo, hi]."""
    n = rng.randint(lo, hi)
    if n == 1:
        return rng.randint(0, 9)
    return rng.randint(10 ** (n - 1), 10**n - 1)


def _gen_examples(
    n: int, lo: int, hi: int, reverse_output: bool, rng: random.Random
) -> list[AdditionExample]:
    seen: set[tuple[int, int]] = set()
    examples: list[AdditionExample] = []
    attempts = 0
    max_attempts = n * 50
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        a = _sample_operand(rng, lo, hi)
        b = _sample_operand(rng, lo, hi)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        result = str(a + b)
        answer = result[::-1] if reverse_output else result
        examples.append(
            AdditionExample(
                a=a,
                b=b,
                prompt=f"{a}+{b}=",
                answer=answer,
                n_digits=max(len(str(a)), len(str(b))),
                has_carry=_has_carry(a, b),
            )
        )
    return examples


class AdditionDataset(Dataset):
    """Dataset autorégressif. Chaque item est padé à ``max_len``.

    Retourne un dict avec ``input_ids``, ``labels``, ``loss_mask`` (1 sur les tokens
    de réponse + EOS), ``prompt_ids``/``prompt_len`` (pour le décodage en éval),
    et des métadonnées (``n_digits``, ``has_carry``, ``target_value``).
    """

    def __init__(
        self,
        examples: list[AdditionExample],
        vocab: AdditionVocab,
        max_len: int,
    ) -> None:
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        vocab = self.vocab
        prompt_ids = vocab.encode(ex.prompt)
        answer_ids = vocab.encode(ex.answer) + [vocab.eos_id]
        full = prompt_ids + answer_ids
        if len(full) > self.max_len:
            raise ValueError(
                f"Séquence ({len(full)}) > max_len ({self.max_len}); augmenter max_len."
            )

        # Décalage autorégressif : prédire le token suivant.
        input_ids = full[:-1]
        labels = full[1:]
        # Masque : 1 sur les positions dont la cible est un token de réponse.
        # Les cibles de réponse occupent les dernières (len(answer_ids)) positions.
        loss_mask = [0] * (len(labels) - len(answer_ids)) + [1] * len(answer_ids)

        pad = self.max_len - 1 - len(input_ids)
        input_ids = input_ids + [vocab.pad_id] * pad
        labels = labels + [vocab.pad_id] * pad
        loss_mask = loss_mask + [0] * pad

        prompt_pad = self.max_len - len(prompt_ids)
        prompt_ids_padded = prompt_ids + [vocab.pad_id] * prompt_pad

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float),
            "prompt_ids": torch.tensor(prompt_ids_padded, dtype=torch.long),
            "prompt_len": torch.tensor(len(prompt_ids), dtype=torch.long),
            "n_digits": torch.tensor(ex.n_digits, dtype=torch.long),
            "has_carry": torch.tensor(int(ex.has_carry), dtype=torch.long),
            "target_value": torch.tensor(ex.a + ex.b, dtype=torch.long),
        }


def make_addition_splits(
    *,
    train_digits: tuple[int, int],
    ood_digits: tuple[int, int],
    n_train: int,
    n_val: int,
    n_test: int,
    n_ood: int,
    reverse_output: bool,
    max_len: int,
    seed: int,
) -> tuple[AdditionVocab, dict[str, AdditionDataset]]:
    """Construit les splits train/val/test (longueurs vues) + ood (longueurs non vues).

    train/val/test partagent la plage ``train_digits`` (dédupliqués entre eux) ;
    ``ood`` utilise ``ood_digits`` strictement disjoint.
    """
    rng = random.Random(seed)
    vocab = AdditionVocab()

    lo, hi = train_digits
    pool = _gen_examples(n_train + n_val + n_test, lo, hi, reverse_output, rng)
    train = pool[:n_train]
    val = pool[n_train : n_train + n_val]
    test = pool[n_train + n_val : n_train + n_val + n_test]

    olo, ohi = ood_digits
    ood = _gen_examples(n_ood, olo, ohi, reverse_output, rng)

    splits = {
        "train": AdditionDataset(train, vocab, max_len),
        "val": AdditionDataset(val, vocab, max_len),
        "test": AdditionDataset(test, vocab, max_len),
        "ood": AdditionDataset(ood, vocab, max_len),
    }
    return vocab, splits
