# colabfit/tools/vast/utils/__init__.py
"""
Utilities package for colabfit.tools.vast

This module provides convenient access to commonly used utility functions
organized by functionality.
"""

# Hashing functions - most commonly needed across modules
from .hashing import _hash, _format_for_hash, config_struct_hash, config_struct_hash_udf

# Data processing functions
from .data_processing import (
    get_last_modified,
    get_date,
    get_pbc_from_cell,
    stringify_df_val_udf,
    str_to_arrayof_str,
    str_to_arrayof_int,
    get_max_string_length,
)

# Schema management
from .schema_management import (
    get_spark_field_type,
    get_stringified_schema,
    _empty_dict_from_schema,
    spark_to_arrow_type,
    spark_schema_to_arrow_schema,
)

from .metadata import _sort_dict, _parse_unstructured_metadata

from .constants import ELEMENT_MAP

from .pyspark_utils import convert_stress

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
    "stringify_df_val_udf",
    "str_to_arrayof_str",
    "str_to_arrayof_int",
    "get_max_string_length",
    # Schema management
    "get_spark_field_type",
    "get_stringified_schema",
    "_empty_dict_from_schema",
    "spark_to_arrow_type",
    "spark_schema_to_arrow_schema",
    # Metadata processing
    "_sort_dict",
    "_parse_unstructured_metadata",
    # Constants
    "ELEMENT_MAP",
    # PySpark utilities
    "convert_stress",
]
