from hashlib import sha512

import numpy as np


def _format_for_hash(v: np.ndarray | list | dict | str | int | float | tuple):
    if isinstance(v, (list, tuple)):
        v = np.array(v)
    if isinstance(v, np.ndarray):
        if np.issubdtype(v.dtype, np.floating):
            return np.round(v.astype(np.float64), decimals=16)
        elif np.issubdtype(v.dtype, np.integer):
            return v.astype(np.int64)
        elif np.issubdtype(v.dtype, np.bool_):
            return v.astype(np.int64)
        else:
            return v
    elif isinstance(v, dict):
        return str(v).encode("utf-8")
    elif isinstance(v, str):
        return v.encode("utf-8")
    elif isinstance(v, (int, float)):
        return np.array(v).data.tobytes()
    else:
        return v


def _hash(
    row: dict, identifying_key_list: list, include_keys_in_hash: bool = False
) -> str:
    identifying_key_list = sorted(identifying_key_list)
    identifiers = [row[k] for k in identifying_key_list]
    _hash = sha512()
    for k, v in zip(identifying_key_list, identifiers):
        # if k in [""]
        if v is None or v == "[]":
            continue
        if include_keys_in_hash:
            _hash.update(_format_for_hash(k))
        _hash.update(_format_for_hash(v))
    return _hash.hexdigest()


def _sorted_struct_hash(
    atomic_numbers: list[int],
    cell: list[float],
    pbc: list[bool],
    positions: list[list[float]],
):
    """Shared structure hashing logic; returns a sha512 digest object."""
    h = sha512()
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
    h.update(bytes(_format_for_hash(sorted_atomic_numbers)))
    h.update(bytes(_format_for_hash(cell)))
    h.update(bytes(_format_for_hash(pbc)))
    h.update(bytes(_format_for_hash(sorted_positions)))
    return h


def config_struct_hash(
    atomic_numbers: list[int],
    cell: list[float],
    pbc: list[bool],
    positions: list[list[float]],
):
    """Structure hashing for configuration creation. Returns hex string."""
    return _sorted_struct_hash(atomic_numbers, cell, pbc, positions).hexdigest()
