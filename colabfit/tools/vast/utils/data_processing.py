import datetime

import dateutil.parser
import numpy as np


def convert_stress(keys: str, stress: list[float]) -> list[list[float]]:
    """Convert a size-6 array of stress components to a 3x3 matrix

    In particular for VASP output. Assumes symmetric matrix.
    Check order of keys."""
    stresses = {k: s for k, s in zip(keys, stress)}
    return [
        [stresses["xx"], stresses["xy"], stresses["xz"]],
        [stresses["xy"], stresses["yy"], stresses["yz"]],
        [stresses["xz"], stresses["yz"], stresses["zz"]],
    ]


def get_last_modified():
    return dateutil.parser.parse(
        datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def get_date():
    return dateutil.parser.parse(
        datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    )


def get_pbc_from_cell(cell: list[list[float]] | np.ndarray) -> list[bool]:
    cell = np.array(cell, dtype=float)
    cell_lengths = np.linalg.norm(cell, axis=1)
    pbc = cell_lengths > 1e-10
    return pbc.astype(bool).tolist()
