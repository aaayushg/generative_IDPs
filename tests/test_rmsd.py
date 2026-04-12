from __future__ import annotations

import numpy as np

from generative_idps.analysis import kabsch_rmsd


def test_kabsch_rmsd_is_zero_for_rotated_points():
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    candidate = reference @ rotation.T

    assert kabsch_rmsd(reference, candidate) < 1e-6
