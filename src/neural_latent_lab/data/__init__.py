"""Générateurs de datasets synthétiques. v0 : addition multi-chiffres."""

from neural_latent_lab.data.addition import (
    AdditionDataset,
    AdditionVocab,
    make_addition_splits,
)

__all__ = ["AdditionDataset", "AdditionVocab", "make_addition_splits"]
