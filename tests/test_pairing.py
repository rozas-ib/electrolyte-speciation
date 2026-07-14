import numpy as np

from speciation import classify_ion_pairing


def test_pairing_classes_are_exclusive_and_complete() -> None:
    free, ssip, cip, agg = classify_ion_pairing(
        np.array([0, 0, 1, 2]), np.array([0, 1, 1, 3])
    )

    assert free.tolist() == [True, False, False, False]
    assert ssip.tolist() == [False, True, False, False]
    assert cip.tolist() == [False, False, True, False]
    assert agg.tolist() == [False, False, False, True]
