from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class AtomRecord:
    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_id: int
    element: str


class PDBTrajectory:
    def __init__(self, atom_records: List[AtomRecord], coordinates: np.ndarray):
        coords = np.asarray(coordinates, dtype=float)
        if coords.ndim != 3 or coords.shape[2] != 3:
            raise ValueError("coordinates must have shape (frames, atoms, 3)")
        if coords.shape[1] != len(atom_records):
            raise ValueError("atom record count does not match coordinates")
        self.atom_records = list(atom_records)
        self.coordinates = coords

    @property
    def n_frames(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.coordinates.shape[1])

    @classmethod
    def from_pdb(cls, path: Union[str, Path]) -> "PDBTrajectory":
        path = Path(path)
        atom_records: List[AtomRecord] = []
        frames: List[np.ndarray] = []
        current_frame: List[List[float]] = []
        current_records: List[AtomRecord] = []
        saw_model = False

        def finalize_frame() -> None:
            nonlocal atom_records, current_frame, current_records
            if not current_frame:
                return
            frame = np.asarray(current_frame, dtype=float)
            if not atom_records:
                atom_records = list(current_records)
            elif len(current_records) != len(atom_records):
                raise ValueError(f"Inconsistent atom count in {path}")
            frames.append(frame)
            current_frame = []
            current_records = []

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record_type = line[:6].strip()
                if record_type == "MODEL":
                    saw_model = True
                    finalize_frame()
                    continue
                if record_type in {"ENDMDL", "END"}:
                    finalize_frame()
                    continue
                if record_type not in {"ATOM", "HETATM"}:
                    continue
                current_records.append(
                    AtomRecord(
                        serial=int(line[6:11]),
                        name=line[12:16].strip(),
                        residue_name=line[17:20].strip(),
                        chain_id=line[21].strip() or "A",
                        residue_id=int(line[22:26]),
                        element=(line[76:78].strip() or line[12:16].strip()[0]),
                    )
                )
                current_frame.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])

        finalize_frame()

        if not frames:
            raise ValueError(f"No ATOM records found in {path}")
        if not saw_model:
            frames = [frames[0]]

        return cls(atom_records=atom_records, coordinates=np.stack(frames, axis=0))

    def select_indices(self, atom_names: Sequence[str]) -> np.ndarray:
        atom_name_set = {name.upper() for name in atom_names}
        indices = [index for index, atom in enumerate(self.atom_records) if atom.name.upper() in atom_name_set]
        if not indices:
            raise ValueError(f"No atoms matched selection: {sorted(atom_name_set)}")
        return np.asarray(indices, dtype=int)

    def subset(
        self,
        atom_indices: Optional[Union[np.ndarray, Sequence[int]]] = None,
        frame_indices: Optional[Union[np.ndarray, Sequence[int]]] = None,
    ) -> "PDBTrajectory":
        atom_idx = np.arange(self.n_atoms) if atom_indices is None else np.asarray(atom_indices, dtype=int)
        frame_idx = np.arange(self.n_frames) if frame_indices is None else np.asarray(frame_indices, dtype=int)
        atom_records = [self.atom_records[index] for index in atom_idx.tolist()]
        return PDBTrajectory(atom_records=atom_records, coordinates=self.coordinates[np.ix_(frame_idx, atom_idx)])

    def flatten(self) -> np.ndarray:
        return self.coordinates.reshape(self.n_frames, self.n_atoms * 3)

    def write_pdb(self, path: Union[str, Path]) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as handle:
            for frame_index, frame in enumerate(self.coordinates, start=1):
                handle.write(f"MODEL     {frame_index}\n")
                for atom, xyz in zip(self.atom_records, frame):
                    handle.write(
                        "ATOM  {serial:5d} {name:<4}{residue_name:>4} {chain_id:1}{residue_id:4d}    "
                        "{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2}\n".format(
                            serial=atom.serial,
                            name=atom.name,
                            residue_name=atom.residue_name,
                            chain_id=atom.chain_id,
                            residue_id=atom.residue_id,
                            x=float(xyz[0]),
                            y=float(xyz[1]),
                            z=float(xyz[2]),
                            occupancy=1.0,
                            temp_factor=0.0,
                            element=atom.element[:2],
                        )
                    )
                handle.write("ENDMDL\n")
            handle.write("END\n")
