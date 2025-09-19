import json
import logging
import os
import sys
from hashlib import sha512
import dateutil.parser
import datetime

import numpy as np
import pyarrow as pa
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    ArrayType,
    TimestampType,
)

logger = logging.getLogger(__name__)


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


def get_spark_field_type(schema: StructType, field_name: str) -> DataType:
    for field in schema:
        if field.name == "$row_id":
            return LongType()
        if field.name == field_name:
            return field.dataType
    raise ValueError(f"Field name {field_name} not found in schema")


def get_stringified_schema(schema: StructType) -> StructType:
    new_fields = []
    for field in schema:
        if field.dataType.typeName() == "array":
            new_fields.append(StructField(field.name, StringType(), field.nullable))
        else:
            new_fields.append(field)
    return StructType(new_fields)


def spark_to_arrow_type(spark_type: DataType) -> pa.DataType:
    """
    Convert PySpark type to PyArrow type.
    Do not include field.nullable, as this conflicts with vastdb-sdk
    """
    if isinstance(spark_type, IntegerType):
        return pa.int32()
    elif isinstance(spark_type, LongType):
        return pa.int64()
    elif isinstance(spark_type, DoubleType):
        return pa.float64()
    elif isinstance(spark_type, StringType):
        return pa.string()
    elif isinstance(spark_type, TimestampType):
        return pa.timestamp("us")
    elif isinstance(spark_type, BooleanType):
        return pa.bool_()
    elif isinstance(spark_type, ArrayType):
        element_type = spark_type.elementType
        if isinstance(element_type, StringType):
            return pa.list_(pa.string())
        elif isinstance(element_type, BooleanType):
            return pa.list_(pa.bool_())
        elif isinstance(element_type, DoubleType):
            return pa.list_(pa.float64())
        elif isinstance(element_type, IntegerType):
            return pa.list_(pa.int32())
        elif isinstance(element_type, ArrayType):
            return pa.list_(spark_to_arrow_type(element_type))
        else:
            raise ValueError(f"Unsupported array element type: {element_type}")
    elif isinstance(spark_type, StructType):
        return pa.schema(
            [
                pa.field(field.name, spark_to_arrow_type(field.dataType))
                for field in spark_type
            ]
        )
    else:
        raise ValueError(f"Unsupported type: {spark_type}")


def spark_schema_to_arrow_schema(spark_schema: StructType) -> pa.Schema:
    """
    Convert PySpark schema to a PyArrow Schema.
    """
    fields = []
    for field in spark_schema:
        if field.name == "$row_id":
            fields.append(pa.field(field.name, pa.uint64()))
        else:
            fields.append(pa.field(field.name, spark_to_arrow_type(field.dataType)))
    arrow_schema = pa.schema(fields)
    for field in arrow_schema:
        field = field.with_nullable(True)
    return arrow_schema


def _empty_dict_from_schema(schema: StructType) -> dict:
    empty_dict = {}
    for field in schema:
        empty_dict[field.name] = None
    return empty_dict


def _sort_dict(dictionary: dict) -> dict:
    keys = list(dictionary.keys())
    keys.sort()
    return {k: dictionary[k] for k in keys}


def _parse_unstructured_metadata(md_json: dict) -> dict:
    if md_json == {}:
        return {
            "metadata": None,
            "metadata_id": None,
            "metadata_path": None,
            "metadata_size": None,
        }
    md = {}
    for key, val in md_json.items():
        if key in ["_id", "hash", "colabfit-id", "last_modified", "software", "method"]:
            continue
        if isinstance(val, dict):
            if "source-value" in val.keys():
                val = val["source-value"]
        if isinstance(val, list) and len(val) == 1:
            val = val[0]
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, dict):
            val = _sort_dict(val)
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        md[key] = val
    md = _sort_dict(md)
    md_hash = str(_hash(md, md.keys(), include_keys_in_hash=True))
    md["hash"] = md_hash
    md["id"] = f"MD_{md_hash[:25]}"
    split = md["id"][-4:]
    filename = f"{md['id']}.json"
    after_bucket = os.path.join(split, filename)
    metadata = json.dumps(md)
    return {
        "metadata": metadata,
        "metadata_id": md["id"],
        "metadata_path": after_bucket,
        "metadata_size": sys.getsizeof(metadata),
    }


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


############################################################
# Functions for converting column values to string
############################################################


@sf.udf(returnType=StringType())
def stringify_df_val_udf(val):
    if val is not None:
        return str(val)


#####################################################################
# Functions for splitting and combining oversize array column values
#####################################################################
def get_max_string_length(df: DataFrame, column_name: str) -> int:

    max_len = (
        df.select(sf.length(column_name).alias("string_length"))
        .agg(sf.max("string_length"))
        .collect()[0][0]
    )
    if max_len is None:
        return 0
    return max_len


ELEMENT_MAP = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    113: "Nh",
    114: "Fl",
    115: "Mc",
    116: "Lv",
    117: "Ts",
    118: "Og",
}
