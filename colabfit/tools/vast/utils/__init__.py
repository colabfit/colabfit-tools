# colabfit/tools/vast/utils/__init__.py
"""
Utilities package for colabfit.tools.vast

This module provides convenient access to commonly used utility functions
organized by functionality.
"""

from .constants import ELEMENT_MAP

# Data processing functions
from .data_processing import (
    generate_ds_id,
    get_date,
    get_last_modified,
    get_pbc_from_cell,
    convert_stress,
)

# Hashing functions - most commonly needed across modules
from .hashing import _format_for_hash, _hash, config_struct_hash, config_struct_hash_udf
from .metadata import _parse_unstructured_metadata, _sort_dict
from .pyspark_utils import (
    get_max_string_length,
    str_to_arrayof_bool,
    str_to_arrayof_double,
    str_to_arrayof_int,
    str_to_arrayof_str,
    str_to_arrayofarrayof_double,
    stringify_df_val_udf,
)

# Schema management
from .schema_management import (
    _empty_dict_from_schema,
    get_spark_field_type,
    get_stringified_schema,
    spark_schema_to_arrow_schema,
    spark_to_arrow_type,
)


# Vast utils
from .vast_utils import append_wip_table_to_prod, get_session

__all__ = [
    # Hashing
    "_hash",
    "_format_for_hash",
    "config_struct_hash",
    "config_struct_hash_udf",
    # Data processing
    "get_last_modified",
    "get_date",
    "get_pbc_from_cell",
    "generate_ds_id",
    # Schema management
    "get_spark_field_type",
    "get_stringified_schema",
    "_empty_dict_from_schema",
    "spark_to_arrow_type",
    "spark_schema_to_arrow_schema",
    # PySpark utilities
    "stringify_df_val_udf",
    "str_to_arrayof_str",
    "str_to_arrayof_int",
    "str_to_arrayof_double",
    "str_to_arrayof_bool",
    "str_to_arrayofarrayof_double",
    "get_max_string_length",
    # Metadata processing
    "_sort_dict",
    "_parse_unstructured_metadata",
    # Constants
    "ELEMENT_MAP",
    # PySpark utilities
    "convert_stress",
    # Vast utils
    "get_session",
    "append_wip_table_to_prod",
]
