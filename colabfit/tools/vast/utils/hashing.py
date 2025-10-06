from hashlib import sha512

import numpy as np
from pyspark.sql import functions as sf
from pyspark.sql.types import StringType


def _format_for_hash(v: np.ndarray | list | dict | str | int | float | tuple):
    if isinstance(v, np.ndarray):
        if np.issubdtype(v.dtype, np.floating):
            return np.round(v.astype(np.float64), decimals=16)
        elif np.issubdtype(v.dtype, np.integer):
            return v.astype(np.int64)
        elif np.issubdtype(v.dtype, bool):
            return v.astype(np.int64)
        else:
            return v
    elif isinstance(v, list):
        return np.array(v).data.tobytes()
    elif isinstance(v, dict):
        return str(v).encode("utf-8")
    elif isinstance(v, str):
        return v.encode("utf-8")
    elif isinstance(v, (int, float)):
        return np.array(v).data.tobytes()
    elif isinstance(v, tuple):
        return np.array(v).data.tobytes()
    else:
        return v


def _hash(
    row: list, identifying_key_list: list, include_keys_in_hash: bool = False
) -> int:
    identifying_key_list = sorted(identifying_key_list)
    identifiers = [row[i] for i in identifying_key_list]
    _hash = sha512()
    for k, v in zip(identifying_key_list, identifiers):
        if v is None or v == "[]":
            continue
        if include_keys_in_hash:
            _hash.update(_format_for_hash(k))
        _hash.update(_format_for_hash(v))
    return int(_hash.hexdigest(), 16)


def config_struct_hash(
    atomic_numbers: list[int],
    cell: list[float],
    pbc: list[bool],
    positions: list[list[float]],
):
    """Structure hashing for configuration creation"""
    _hash = sha512()
    positions = np.array(positions)
    sort_ixs = np.lexsort(
        (
            positions[:, 2],
            positions[:, 1],
            positions[:, 0],
        )
    )
    sorted_positions = positions[sort_ixs]
    atomic_numbers = np.array(atomic_numbers)
    sorted_atomic_numbers = atomic_numbers[sort_ixs]
    _hash.update(bytes(_format_for_hash(sorted_atomic_numbers)))
    _hash.update(bytes(_format_for_hash(cell)))
    _hash.update(bytes(_format_for_hash(pbc)))
    _hash.update(bytes(_format_for_hash(sorted_positions)))
    return int(_hash.hexdigest(), 16)


@sf.udf(returnType=StringType())
def config_struct_hash_udf(
    atomic_numbers: list[int],
    cell: list[float],
    pbc: list[bool],
    *positions: list[float],
):
    """
    Will hash in the following order: atomic_numbers, cell, pbc, positions

    Position columns will be concatenated and sorted by x, y, z
    Atomic numbers will be sorted by the position sorting
    """
    _hash = sha512()
    positions = []
    for v in positions:
        if v is None or v == "[]":
            continue
        else:
            positions.extend(v)
    positions = np.array(positions)
    sort_ixs = np.lexsort(
        (
            positions[:, 2],
            positions[:, 1],
            positions[:, 0],
        )
    )
    sorted_positions = positions[sort_ixs]
    atomic_numbers = np.array(atomic_numbers)
    sorted_atomic_numbers = atomic_numbers[sort_ixs]
    _hash.update(bytes(_format_for_hash(sorted_atomic_numbers)))
    _hash.update(bytes(_format_for_hash(cell)))
    _hash.update(bytes(_format_for_hash(pbc)))
    _hash.update(bytes(_format_for_hash(sorted_positions)))
    return str(int(_hash.hexdigest(), 16))
