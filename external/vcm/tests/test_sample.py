import vcm


def test_train_test_split_sample_ensure_deterministic(regtest):
    seq = list(range(100))
    out = vcm.train_test_split_sample(seq, 5, train_samples=3, test_samples=2, seed=0)
    print(out, file=regtest)


def test_train_test_split_sample_invariant_to_unsorted(regtest):
    seq1 = [0, 1, 2, 4]
    seq2 = [0, 2, 1, 4]
    out1 = vcm.train_test_split_sample(seq1, 3, train_samples=1, test_samples=1, seed=0)
    out2 = vcm.train_test_split_sample(seq2, 3, train_samples=1, test_samples=1, seed=0)

    assert out1 == out2
