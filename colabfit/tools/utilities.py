import json
import os
import sys
from ast import literal_eval
from functools import partial
from hashlib import sha512
from pathlib import Path
from types import NoneType

import numpy as np
import pyarrow as pa
from pyspark.sql import Row
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructType,
    TimestampType,
)

BUCKET_DIR = "/vast/gw2338/METADATA"


def _format_for_hash(v):
    if isinstance(v, np.ndarray):
        if np.issubdtype(v.dtype, np.floating):
            return np.round_(v.astype(np.float64), decimals=16)
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
    else:
        return v


def _hash(row, indentifying_key_list, include_keys_in_hash=False):
    identifiers = [row[i] for i in indentifying_key_list]
    # sort_for_hash = ["positions", "atomic_forces"]
    _hash = sha512()
    for k, v in zip(indentifying_key_list, identifiers):
        if v is None:
            continue
        # elif k in sort_for_hash:
        #     v = np.array(v)
        #     sorted_v = v[np.lexsort((
        #         v[:, 2],
        #         v[:, 1],
        #         v[:, 0],
        #     ))]
        #     _hash.update(bytes(_format_for_hash(sorted_v)))
        else:
            if include_keys_in_hash:
                _hash.update(bytes(_format_for_hash(k)))
            _hash.update(bytes(_format_for_hash(v)))
    return str(int(_hash.hexdigest(), 16))


# def _hash(row, indentifying_fields_list):
#     identifiers = [row[i] for i in indentifying_fields_list]
#     sort_for_hash = ["positions"]
#     _hash = sha512()
#     for k, v in zip(indentifying_fields_list, identifiers):
#         if v is None:
#             continue
#         elif k in sort_for_hash:
#             v = np.array(v)
#             sorted_v = v[
#                 np.lexsort(
#                     (
#                         v[:, 2],
#                         v[:, 1],
#                         v[:, 0],
#                     )
#                 )
#             ]
#             _hash.update(bytes(_format_for_hash(sorted_v)))
#         else:
#             _hash.update(bytes(_format_for_hash(v)))
#     return int(_hash.hexdigest(), 16)


def get_spark_field_type(schema, field_name):
    for field in schema:
        if field.name == field_name:
            return field.dataType
    raise ValueError(f"Field name {field_name} not found in schema")


def spark_to_arrow_type(spark_type):
    """
    Convert PySpark type to PyArrow type.
    Do not include field.nullable, as this conflicts with vastdb-sdk
    """
    if isinstance(spark_type, IntegerType):
        return pa.int32()
    elif isinstance(spark_type, StringType):
        return pa.string()
    elif isinstance(spark_type, TimestampType):
        return pa.timestamp("ns")
    elif isinstance(spark_type, StructType):
        return pa.struct(
            [
                pa.field(field.name, spark_to_arrow_type(field.dataType))
                for field in spark_type
            ]
        )
    else:
        raise ValueError(f"Unsupported type: {spark_type}")


def spark_schema_to_arrow_schema(spark_schema):
    """
    Convert PySpark schema to a PyArrow Schema.
    """
    fields = []
    for field in spark_schema:
        if field.name == "$row_id":
            fields.append(pa.field(field.name, pa.uint64()))
        else:
            fields.append(pa.field(field.name, spark_to_arrow_type(field.dataType)))
    return pa.schema(fields)


def arrow_record_batch_to_rdd(schema, batch):
    names = schema.fieldNames()
    arrays = [batch.column(i) for i in range(batch.num_columns)]
    for i in range(batch.num_rows):
        yield {names[j]: array[i].as_py() for j, array in enumerate(arrays)}


def _empty_dict_from_schema(schema):
    empty_dict = {}
    for field in schema:
        empty_dict[field.name] = None
    return empty_dict


def _sort_dict(dictionary):
    keys = list(dictionary.keys())
    keys.sort()
    return {k: dictionary[k] for k in keys}


def _parse_unstructured_metadata(md_json):
    md = {}
    for key, val in md_json.items():
        if key in ["_id", "hash", "colabfit-id", "last_modified", "software", "method"]:
            continue
        if "source-value" in val.keys():
            source_value = val["source-value"]
        else:
            source_value = val
        if isinstance(source_value, list) and len(source_value) == 1:
            source_value = source_value[0]
        if isinstance(source_value, dict):
            source_value = _sort_dict(source_value)
        if isinstance(source_value, bytes):
            source_value = source_value.decode("utf-8")
        md[key] = source_value
    md = _sort_dict(md)
    md_hash = _hash(md, md.keys(), include_keys_in_hash=True)
    md["hash"] = md_hash
    md["id"] = f"MD_{md_hash[:25]}"
    split = md["id"][-4:]
    filename = f"{md['id']}.json"
    full_path = os.path.join(BUCKET_DIR, "MD", split, filename)
    if not os.path.isfile(full_path):
        # Write iff the ID is new and unique
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(md, f)
    return {
        "metadata_id": md["id"],
        "metadata_path": full_path,
        "metadata_size": sys.getsizeof(json.dumps(md)),
    }


def stringify_lists(row_dict):
    """
    Replace list/tuple fields with comma-separated strings.
    Spark and Vast both support array columns, but the connector does not,
    so keeping cell values in list format crashes the table.
    Use with dicts
    TODO: Remove when no longer necessary
    """
    for key, val in row_dict.items():
        if isinstance(val, (list, tuple, dict)):
            row_dict[key] = str(val)
        # Below would convert numpy arrays to comma-separated
        elif isinstance(val, np.ndarray):
            row_dict[key] = str(val.tolist())
    return row_dict


def stringify_rows(row):
    """
    Convert list/tuple fields to comma-separated strings.
    Use with spark Rows
    Should be mapped as DataFrame.rdd.map(stringify_rows)"""
    row_dict = row.asDict()
    for key, val in row_dict.items():
        if isinstance(val, (list, tuple, dict)):
            row_dict[key] = str(val)
    new_row = Row(**row_dict)
    return new_row


def stringify_rows_to_dict(row):
    """
    Convert list/tuple fields to comma-separated strings.
    Use with spark Rows
    Should be mapped as DataFrame.rdd.map(stringify_rows)"""
    row_dict = row.asDict()
    for key, val in row_dict.items():
        if isinstance(val, (list, tuple, dict)):
            row_dict[key] = str(val)
    return row_dict


def stringify_row_dict(row_dict):
    for key, val in row_dict.items():
        if isinstance(val, (list, tuple, dict)):
            row_dict[key] = str(val)
    return row_dict


def unstringify(row):
    """Should be mapped as DataFrame.rdd.map(unstringify)"""
    row_dict = row.asDict()
    try:
        for key, val in row_dict.items():
            if key == "metadata":
                continue
            elif isinstance(val, str) and len(val) > 0 and val[0] in ["["]:
                dval = literal_eval(row[key])
                row_dict[key] = dval
            else:
                continue
        new_row = Row(**row_dict)
    except Exception as e:
        print(e)
        print(row_dict)
    return new_row


def unstring_df_val(val):
    if isinstance(val, str) and len(val) > 0 and val[0] in ["["]:
        dval = literal_eval(val)
        return dval
    else:
        return val


def stringify_df_val(val):
    if isinstance(val, (list, tuple, dict)):
        return str(val)
    else:
        return val


def unstringify_row_dict(row_dict):
    """Should be mapped as rdd.map(unstringify_row_dict)"""
    for key, val in row_dict.items():
        if key == "metadata":
            continue
        if isinstance(val, str) and len(val) > 0 and val[0] in ["{", "["]:
            dval = literal_eval(row_dict[key])
            row_dict[key] = dval
    return row_dict


def append_ith_element_to_rdd_labels(row_elem):
    """
    row_elem: tuple created by joining two RDD.zipWithIndex
    new_labels: list of labels
    """
    (index, (co_row, new_labels)) = row_elem
    val = co_row.get("labels")
    if val is None:
        val = new_labels
    else:
        val.extend(new_labels)
        val = list(set(val))
    co_row["labels"] = val
    return co_row


def add_elem_to_row_dict(col, elem, row_dict):
    val = row_dict.get(col)
    if val is None:
        val = [elem]
    else:
        val.append(elem)
        val = list(set(val))
    row_dict[col] = val
    return row_dict


##########################################################
# Functions for splitting oversize arrays to columns
##########################################################


def split_arr_map(column, row_dict):
    n = 60000
    col_string = "".join(
        np.array2string(
            np.array(row_dict[column]), threshold=np.inf, separator=","
        ).replace("\n", "")
    )
    row_dict[column] = [col_string[i : i + n] for i in range(0, len(col_string), n)]
    return row_dict


def stacked_arrays_to_columns(column, max_chunks, row_dict):
    empty = max_chunks - len(row_dict[column])
    full = len(row_dict[column])
    for i, force_arr in enumerate(row_dict[column]):
        row_dict[f"{column}_{i:02d}"] = force_arr
    # row_dict[column] = row_dict.pop(f"{column}_1")
    for i in range(empty):
        row_dict[f"{column}_{(full+i+1):02d}"] = None
    return row_dict


def split_size_n_arrs_to_cols(rdd, column):
    rdd = rdd.map(partial(split_arr_map, column))
    max_chunks = rdd.map(lambda x: len(x[column])).max()
    rdd = rdd.map(partial(stacked_arrays_to_columns, column, max_chunks))
    return rdd


# def update_schema(rdd, schema):
#     """Use with array splitting functions to add string columns to schema"""
#     keys = list(rdd.map(lambda x: x.keys()).take(1)[0])
#     extra_keys = [x for x in keys if x not in schema.fieldNames()]
#     for key in extra_keys:
#         schema.add(StructField(key, StringType(), True))
#     return schema


##########################################################
# Functions for writing values to files
##########################################################


def _write_value(path_prefix, id_str, filetype, BUCKET_DIR, value):
    """i.e.: _write_value(
    value=co['positions'],
    'CO/positions', co['id'],
    'txt', '/save/here'
    )
    """
    # Use the final 4 digits of the id for an ~1000-way split
    split = id_str[-4:]
    filename = f"{id_str}.{filetype}"
    full_path = Path(BUCKET_DIR) / path_prefix / split / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(str(value))
    return full_path


def write_value_to_file(path_prefix, extension, BUCKET_DIR, write_column, row):
    """i.e.: partial(_write_value(
    'CO/positions',
    'txt',
    '/save/here'
    'positions',
    )
    """
    id = row["id"]
    # Use the final 4 digits of the id for an ~1000-way split
    value = row[write_column]
    split = id[-4:]
    filename = f"{id}.{extension}"
    full_path = Path(BUCKET_DIR) / path_prefix / split / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(str(value))
    row_dict = row.asDict()
    row_dict[write_column] = str(full_path)
    return Row(**row_dict)


def multi_value_to_file(path_prefixes, extension, BUCKET_DIR, write_columns, row):
    """i.e.: partial(_write_value(
    'CO/positions',
    'txt',
    '/save/here'
    'positions',
    )
    """
    id = row["id"]
    split = id[-4:]

    row_dict = row.asDict()
    for write_column, path_prefix in zip(write_columns, path_prefixes):
        value = row[write_column]
        filename = f"{id}.{extension}"
        full_path = Path(BUCKET_DIR) / path_prefix / split / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(str(value))
        row_dict[write_column] = str(full_path)
    return Row(**row_dict)


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
