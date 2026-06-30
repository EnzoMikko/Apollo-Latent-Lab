from neural_latent_lab.data.addition import (
    AdditionVocab,
    _has_carry,
    make_addition_splits,
)


def test_vocab_roundtrip():
    vocab = AdditionVocab()
    assert vocab.decode(vocab.encode("12+45=")) == "12+45="


def test_has_carry():
    assert _has_carry(5, 5) is True       # 5+5=10
    assert _has_carry(12, 34) is False    # 46, pas de retenue
    assert _has_carry(19, 1) is True      # 20


def test_reverse_output_round_trip():
    vocab, splits = make_addition_splits(
        train_digits=(1, 2), ood_digits=(3, 4),
        n_train=50, n_val=10, n_test=10, n_ood=10,
        reverse_output=True, max_len=32, seed=0,
    )
    ex = splits["train"].examples[0]
    # answer est le résultat inversé ; le ré-inverser redonne a+b.
    assert int(ex.answer[::-1]) == ex.a + ex.b


def test_ood_lengths_disjoint():
    vocab, splits = make_addition_splits(
        train_digits=(1, 3), ood_digits=(4, 6),
        n_train=200, n_val=20, n_test=20, n_ood=50,
        reverse_output=True, max_len=48, seed=1,
    )
    train_lengths = {e.n_digits for e in splits["train"].examples}
    ood_lengths = {e.n_digits for e in splits["ood"].examples}
    assert max(train_lengths) <= 3
    assert min(ood_lengths) >= 4
    assert train_lengths.isdisjoint(ood_lengths)


def test_item_shapes_and_mask():
    vocab, splits = make_addition_splits(
        train_digits=(1, 2), ood_digits=(3, 4),
        n_train=10, n_val=2, n_test=2, n_ood=2,
        reverse_output=True, max_len=24, seed=0,
    )
    item = splits["train"][0]
    assert item["input_ids"].shape[0] == 23      # max_len - 1
    assert item["labels"].shape == item["input_ids"].shape
    assert item["loss_mask"].shape == item["labels"].shape
    # le masque ne couvre que des positions de réponse (non vides).
    assert item["loss_mask"].sum() > 0
