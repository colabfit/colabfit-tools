import json
import os
import sys
from ast import literal_eval
from hashlib import sha512

import numpy as np
import pyarrow as pa
from pyspark.sql import Row
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


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
    elif isinstance(v, tuple):
        return np.array(v).data.tobytes()
    else:
        return v


def _hash(row, identifying_key_list, include_keys_in_hash=False):
    identifying_key_list = sorted(identifying_key_list)
    identifiers = [row[i] for i in identifying_key_list]
    # sort_for_hash = ["positions", "atomic_forces"]
    _hash = sha512()
    for k, v in zip(identifying_key_list, identifiers):
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
    return int(_hash.hexdigest(), 16)


def get_spark_field_type(schema, field_name):
    for field in schema:
        if field.name == field_name:
            return field.dataType
    raise ValueError(f"Field name {field_name} not found in schema")


def get_stringified_schema(schema):
    new_fields = []
    for field in schema:
        if field.dataType.typeName() == "array":
            new_fields.append(StructField(field.name, StringType(), field.nullable))
        else:
            new_fields.append(field)
    return StructType(new_fields)


def spark_to_arrow_type(spark_type):
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
    elif isinstance(spark_type, StructType):
        return pa.schema(
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


# if not os.path.isfile(full_path):
#         # Write iff the ID is new and unique
#         os.makedirs(os.path.dirname(full_path), exist_ok=True)
#         with open(full_path, "w") as f:
#             json.dump(md, f)


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


def convert_stress(keys, stress):
    """Convert a size-6 array of stress components to a 3x3 matrix

    In particular for VASP output. Assumes symmetric matrix.
    Check order of keys."""
    stresses = {k: s for k, s in zip(keys, stress)}
    return [
        [stresses["xx"], stresses["xy"], stresses["xz"]],
        [stresses["xy"], stresses["yy"], stresses["yz"]],
        [stresses["xz"], stresses["yz"], stresses["zz"]],
    ]


##########################################################
# Functions for splitting oversize arrays to columns
##########################################################
def get_max_string_length(df, column_name):

    max_len = (
        df.select(column_name)
        .select(sf.length(column_name).alias("string_length"))
        .agg(sf.max("string_length"))
        .collect()[0][0]
    )
    if max_len is None:
        return 0
    return max_len


def split_long_string_cols(df, column_name: str, max_string_length: int):
    """
    Splits a long string column into multiple columns based on a maximum string length.
    :param df: Input DataFrame with array cols already stringified
    :param column_name: Name of the column containing the long string
    :param max_string_length: Maximum length for each split string
    :return: DataFrame with the long string split across multiple columns
    """
    print(f"Column split: {column_name}")
    overflow_columns = [
        f"{'_'.join(column_name.split('_')[:-1])}_{i + 1:02}" for i in range(19)
    ]
    if not all([col in df.columns for col in overflow_columns]):
        raise ValueError("Overflow columns not found in target DataFrame schema")
    if get_max_string_length(df, column_name) <= max_string_length:
        for col in overflow_columns:
            df = df.withColumn(col, sf.lit("[]").cast(StringType()))
        return df
    all_columns = [column_name] + overflow_columns
    tmp_columns = [f"{col_name}_tmp" for col_name in all_columns]
    df = df.withColumn("total_length", sf.length(sf.col(column_name)))
    max_string_length = 35000 // 15
    substring_exprs = [
        sf.when(
            sf.length(sf.col(column_name)) - (i * max_string_length) > 0,
            sf.substring(
                sf.col(column_name), (i * max_string_length + 1), max_string_length
            ),
        )
        .otherwise(sf.lit("[]"))
        .alias(col_name)
        for i, col_name in enumerate(tmp_columns)
    ]
    df = df.select("*", *substring_exprs)
    for tmp_col, col in zip(tmp_columns, all_columns):
        df = df.drop(col).withColumnRenamed(f"{tmp_col}", col)
    df = df.drop("total_length")
    return df


# ##########################################################
# # Functions for writing values to files
# ##########################################################


# def _write_value(path_prefix, id_str, filetype, BUCKET_DIR, value):
#     """i.e.: _write_value(
#     value=co['positions'],
#     'CO/positions', co['id'],
#     'txt', '/save/here'
#     )
#     """
#     # Use the final 4 digits of the id for an ~1000-way split
#     split = id_str[-4:]
#     filename = f"{id_str}.{filetype}"
#     full_path = Path(BUCKET_DIR) / path_prefix / split / filename
#     full_path.parent.mkdir(parents=True, exist_ok=True)
#     full_path.write_text(str(value))
#     return full_path


# def write_value_to_file(path_prefix, extension, BUCKET_DIR, write_column, row):
#     """i.e.: partial(_write_value(
#     'CO/positions',
#     'txt',
#     '/save/here'
#     'positions',
#     )
#     """
#     id = row["id"]
#     # Use the final 4 digits of the id for an ~1000-way split
#     value = row[write_column]
#     split = id[-4:]
#     filename = f"{id}.{extension}"
#     full_path = Path(BUCKET_DIR) / path_prefix / split / filename
#     full_path.parent.mkdir(parents=True, exist_ok=True)
#     full_path.write_text(str(value))
#     row_dict = row.asDict()
#     row_dict[write_column] = str(full_path)
#     return Row(**row_dict)


# def multi_value_to_file(path_prefixes, extension, BUCKET_DIR, write_columns, row):
#     """i.e.: partial(_write_value(
#     'CO/positions',
#     'txt',
#     '/save/here'
#     'positions',
#     )
#     """
#     id = row["id"]
#     split = id[-4:]

#     row_dict = row.asDict()
#     for write_column, path_prefix in zip(write_columns, path_prefixes):
#         value = row[write_column]
#         filename = f"{id}.{extension}"
#         full_path = Path(BUCKET_DIR) / path_prefix / split / filename
#         full_path.parent.mkdir(parents=True, exist_ok=True)
#         full_path.write_text(str(value))
#         row_dict[write_column] = str(full_path)
#     return Row(**row_dict)


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
