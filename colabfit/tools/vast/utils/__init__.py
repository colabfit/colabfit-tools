# colabfit/tools/vast/utils/__init__.py
"""
Utilities package for colabfit.tools.vast

This module provides convenient access to commonly used utility functions
organized by functionality.
"""

from .constants import ELEMENT_MAP

# Data processing functions
from .data_processing import (
    convert_stress,
    generate_ds_id,
    get_date,
    get_last_modified,
    get_pbc_from_cell,
)

# Hashing functions - most commonly needed across modules
from .hashing import (
    _format_for_hash,
    _new_hash,
    new_config_struct_hash,
)
from .metadata import _parse_unstructured_metadata, _sort_dict
from .arrow_utils import get_max_string_length

# Schema management
from .schema_management import _empty_dict_from_schema

# Vast utils
from .vast_utils import append_wip_table_to_prod, get_session

__all__ = [
    # Hashing
    "_new_hash",
    "_format_for_hash",
    "new_config_struct_hash",
    # Data processing
    "get_last_modified",
    "get_date",
    "get_pbc_from_cell",
    "generate_ds_id",
    # Schema management
    "_empty_dict_from_schema",
    # Arrow utilities
    "get_max_string_length",
    # Metadata processing
    "_sort_dict",
    "_parse_unstructured_metadata",
    # Constants
    "ELEMENT_MAP",
    # Data processing
    "convert_stress",
    # Vast utils
    "get_session",
    "append_wip_table_to_prod",
]
