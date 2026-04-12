from __future__ import annotations

from generative_idps.trajectory import PDBTrajectory


def test_multi_model_pdb_parsing(tmp_path):
    pdb_path = tmp_path / "toy.pdb"
    pdb_path.write_text(
        "MODEL     1\n"
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C   GLY A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ENDMDL\n"
        "MODEL     2\n"
        "ATOM      1  CA  GLY A   1       0.100   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  C   GLY A   1       1.100   0.000   0.000  1.00  0.00           C\n"
        "ENDMDL\n"
        "END\n",
        encoding="utf-8",
    )

    trajectory = PDBTrajectory.from_pdb(pdb_path)

    assert trajectory.n_frames == 2
    assert trajectory.n_atoms == 2
    assert trajectory.flatten().shape == (2, 6)
    assert trajectory.select_indices(["CA"]).tolist() == [0]
