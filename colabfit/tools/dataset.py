import datetime
import itertools
import json
import os
import tempfile
import warnings
from copy import deepcopy
from hashlib import sha512

import dateutil
import numpy as np
import pyspark.sql.functions as sf
from ase.units import create_units
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from unidecode import unidecode

from colabfit import MAX_STRING_LENGTH

dataset_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", LongType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("nconfigurations", IntegerType(), True),
        StructField("nsites", IntegerType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("elements", StringType(), True),  # ArrayType(StringType())
        StructField(
            "total_elements_ratios", StringType(), True
        ),  # ArrayType(DoubleType())
        StructField(
            "nperiodic_dimensions", StringType(), True
        ),  # ArrayType(IntegerType())
        StructField(
            "dimension_types", StringType(), True
        ),  # ArrayType(ArrayType(IntegerType()))
        StructField("atomization_energy_count", IntegerType(), True),
        StructField("adsorption_energy_count", IntegerType(), True),
        StructField("free_energy_count", IntegerType(), True),
        StructField("potential_energy_count", IntegerType(), True),
        StructField("atomic_forces_count", IntegerType(), True),
        StructField("band_gap_count", IntegerType(), True),
        StructField("cauchy_stress_count", IntegerType(), True),
        StructField("authors", StringType(), True),  # ArrayType(StringType())
        StructField("description", StringType(), True),
        StructField("extended_id", StringType(), True),
        StructField("license", StringType(), True),
        StructField("publication_link", StringType(), True),  # ArrayType(StringType())
        StructField("data_link", StringType(), True),  # ArrayType(StringType()
        StructField("other_links", StringType(), True),  # ArrayType(StringType()
        StructField("name", StringType(), True),
    ]
)


def _empty_dict_from_schema(schema):
    empty_dict = {}
    for field in schema:
        empty_dict[field.name] = None
    return empty_dict


def stringify_lists(row_dict):
    """
    Replace list/tuple fields with comma-separated strings.
    Spark and Vast both support array columns, but the connector does not,
    so keeping cell values in list format crashes the table.
    TODO: Remove when no longer necessary
    """
    for key, val in row_dict.items():
        if isinstance(val, (list, tuple, dict)):
            row_dict[key] = str(val)
        # Below would convert numpy arrays to comma-separated
        elif isinstance(val, np.ndarray):
            row_dict[key] = str(val.tolist())
    return row_dict


_hash_ignored_fields = ["id", "hash", "last_modified", "extended_id"]


class Dataset:
    """
    A dataset defines a group of configuration sets and computed properties, and
    aggregates information about those configuration sets and properties.

    Attributes:

        configuration_set_ids (list):
            A list of attached configuration sets

        property_ids (list):
            A list of attached properties

        name (str):
            The name of the dataset

        authors (list or str or None):
            The names of the authors of the dataset.

        links (list or str or None):
            External links (e.g., journal articles, Git repositories, ...)
            to be associated with the dataset.

        description (str or None):
            A human-readable description of the dataset.

        aggregated_info (dict):
            A dictionary of information that was aggregated rom all of the
            attached configuration sets and properties. Contains the following
            information:

                From the configuration sets:
                    nconfigurations
                    nsites
                    nelements
                    chemical_systems
                    elements
                    individual_elements_ratios
                    total_elements_ratios
                    configuration_labels
                    configuration_labels_counts
                    chemical_formula_reduced
                    chemical_formula_anonymous
                    chemical_formula_hill
                    nperiodic_dimensions
                    dimension_types

                From the properties:
                    property_types
                    property_fields
                    methods
                    methods_counts
                    property_labels
                    property_labels_counts

        data_license (str):
            License associated with the Dataset's data
    """

    def __init__(
        self,
        name: str,
        authors: list[str],
        publication_link: str,
        data_link: str,
        description: str,
        other_links: list[str] = None,
        ds_id: str = None,
        # configuration_sets: list[str] = None, # not implemented
        data_license: str = "CC-BY-ND-4.0",
    ):
        for auth in authors:
            if not "".join(auth.split(" ")[-1].replace("-", "")).isalpha():
                raise RuntimeError(
                    f"Bad author name '{auth}'. Author names "
                    "can only contain [a-z][A-Z]"
                )

        self.name = name
        self.authors = authors
        self.publication_link = publication_link
        self.data_link = data_link
        self.other_links = other_links
        self.description = description
        self.data_license = data_license
        self.ds_id = ds_id
        self.unique_identifier_kw = [
            k for k in dataset_schema.fieldNames() if k not in _hash_ignored_fields
        ]
        self.spark_row = self.to_spark_row()
        self._hash = hash(self)
        self.id = f"DS_{self._hash}"

        self.spark_row["id"] = self.ds_id
        if ds_id is None:
            raise ValueError("Dataset ID must be provided")
        id_prefix = "_".join(
            [
                self.name,
                "".join([unidecode(auth.split()[-1]) for auth in authors]),
            ]
        )
        if len(id_prefix) > (MAX_STRING_LENGTH - len(ds_id) - 2):
            id_prefix = id_prefix[: MAX_STRING_LENGTH - len(ds_id) - 2]
            warnings.warn(f"ID prefix is too long. Clipping to {id_prefix}")
        extended_id = f"{id_prefix}__{ds_id}"
        self.spark_row["extended_id"] = extended_id
        self.spark_row = stringify_lists(self.spark_row)

    def to_spark_row(self, loader):
        """"""
        if loader.prefix is not None:
            table = f"{loader.prefix}.{loader.config_table}"
        else:
            table = loader.config_table
        config_df = loader.spark.read.jdbc(
            url=loader.url, table=table, properties=loader.properties
        ).where(f"ds_id is {self.ds_id}")

        row_dict = _empty_dict_from_schema(dataset_schema)
        row_dict["last_modified"] = dateutil.parser.parse(
            datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )
        row_dict["nconfigurations"] = len(self.configuration_set_ids)
        row_dict["nsites"] = config_df.agg({"nsites": "sum"}).first()[0]

        row_dict["elements"] = sorted(
            config_df.withColumn(
                "elements_unstrung",
                sf.from_json(sf.col("elements"), sf.ArrayType(sf.StringType())),
            )
            .withColumn("exploded_elements", sf.explode("elements_unstrung"))
            .agg(sf.collect_set("exploded_elements").alias("exploded_elements"))
            .select("exploded_elements")
            .take(1)[0][0]
        )
        row_dict["nelements"] = len(row_dict["elements"])
        atomic = (
            config_df.withColumn(
                "atomic_unstrung",
                sf.from_json(
                    # commented lines use regexp_replace in case of using numpy.ndarray
                    # because str(numpy.ndarray) is not comma-delimited
                    # sf.regexp_replace(
                    #     sf.regexp_replace(
                    sf.col("atomic_numbers"),
                    #           "\[ ", "[").alias(
                    #         "an_first_space_removed"
                    #     ),
                    #     "\s+",
                    #     ", ",
                    # ),
                    sf.ArrayType(IntegerType()),
                ),
            )
            .select("atomic_unstrung")
            .withColumn("exploded_atom", sf.explode("atomic_unstrung"))
            .groupBy(sf.col("exploded_atom").alias("atomic_number"))
            .count()
            .withColumn("ratio", sf.col("count") / row_dict["nsites"])
            .select("ratio", "atomic_number")
            .withColumn(
                "element",
                sf.udf(lambda x: element_map[x], StringType())(sf.col("atomic_number")),
            )
            .select("element", "ratio")
            .collect()
        )
        row_dict["total_elements_ratios"] = dict(
            sorted(atomic, key=lambda x: x["element"])
        )

        row_dict["nperiodic_dimensions"] = config_df.agg(
            sf.collect_set("nperiodic_dimensions")
        ).collect()[0][0]

        row_dict["dimension_types"] = (
            config_df.withColumn(
                "dims_unstrung",
                sf.from_json(sf.col("dimension_types"), sf.ArrayType(sf.StringType())),
            )
            .select("dims_unstrung")
            .agg(sf.collect_set("dims_unstrung"))
            .collect()[0][0]
        )

        prop_df = loader.spark.read.jdbc(
            url=loader.url,
            table="public.property_objects",
            properties=loader.properties,
        ).where(f"dataset_ids is {self.ds_id}")
        for prop in [
            "atomization_energy",
            "adsorption_energy",
            "band_gap",
            "formation_energy",
            "free_energy",
            "potential_energy",
            "atomic_forces",
            "cauchy_stress",
        ]:
            row_dict[f"{prop}_count"] = (
                prop_df.select(prop).where(f"{prop} is not null").count()
            )
        row_dict["nproperty_objects"] = prop_df.count()
        row_dict["nconfigurations"] = config_df.count()

        row_dict["authors"] = str(self.authors)
        row_dict["description"] = self.description
        row_dict["license"] = self.data_license
        row_dict["publication_link"] = self.publication_link
        row_dict["data_link"] = self.data_link
        if self.other_links is not None:
            row_dict["other_links"] = str(self.other_links)
        row_dict["name"] = self.name

        return row_dict

    @staticmethod
    def _format_for_hash(v):
        if isinstance(v, list):
            return np.array(v).data.tobytes()
        elif isinstance(v, str):
            return v.encode("utf-8")
        elif isinstance(v, (int, float)):
            return np.array(v).data.tobytes()
        else:
            return v

    def __hash__(self):
        identifiers = [self.spark_row[i] for i in self.unique_identifier_kw]
        _hash = sha512()
        for k, v in zip(self.unique_identifier_kw, identifiers):

            if v is None:
                continue
            else:
                _hash.update(bytes(k.encode("utf-8")))
                _hash.update(bytes(self._format_for_hash(v)))
        return int(_hash.hexdigest(), 16)

    def __str__(self):
        return (
            f"Dataset(description='{self.description}', "
            # f"nconfiguration_sets={len(self.spark_row['configuration_sets'])}, "
            f"nproperties={self.spark_row['nproperties']})"
            f"nconfigurations={self.spark_row['nconfigurations']}"
        )

    def __repr__(self):
        return str(self)


element_map = {
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
