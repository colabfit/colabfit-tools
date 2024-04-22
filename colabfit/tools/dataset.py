from hashlib import sha512
import datetime
import dateutil
import itertools
import json
import os
import tempfile
import warnings
from copy import deepcopy
from hashlib import sha512
import pyspark.sql.functions as sf
import numpy as np
from ase.units import create_units
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

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
        StructField("links", StringType(), True),  # ArrayType(StringType())
        StructField("name", StringType(), True),
    ]
)


def _empty_dict_from_schema(schema):
    empty_dict = {}
    for field in schema:
        empty_dict[field.name] = None
    return empty_dict


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
        configuration_set_ids,
        property_ids,
        name,
        authors,
        links,
        description,
        aggregated_info,
        data_license="CC-BY-ND-4.0",
    ):
        for auth in authors:
            if not "".join(auth.split(" ")[-1].replace("-", "")).isalpha():
                raise RuntimeError(
                    "Bad author name '{}'. Author names can only contain [a-z][A-Z]".format(
                        auth
                    )
                )

        self.configuration_set_ids = configuration_set_ids
        self.property_ids = property_ids
        self.name = name
        self.authors = authors
        self.links = links
        self.description = description
        self.aggregated_info = aggregated_info
        self.data_license = data_license
        self._hash = hash(self)

    def to_spark_row(self, loader):
        """"""
        if loader.prefix is not None:
            table = f"{loader.prefix}.{loader.config_table}"
        else:
            table = loader.config_table
        df = loader.spark.read.jdbc(
            url=loader.url, table=table, properties=loader.properties
        ).fi

        row_dict = _empty_dict_from_schema(dataset_schema)
        row_dict["last_modified"] = dateutil.parser.parse(
            datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )
        row_dict["nconfigurations"] = len(self.configuration_set_ids)
        row_dict["nsites"] = df.agg({"nsites": "sum"}).first()[0]

        row_dict["elements"] = sorted(
            df.withColumn(
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
            df.withColumn(
                "atomic_unstrung",
                sf.from_json(
                    # Using regexp_replace bc. str(numpy.ndarray) is not comma-delimited
                    sf.regexp_replace(
                        sf.regexp_replace(sf.col("atomic_numbers"), "\[ ", "[").alias(
                            "an_first_space_removed"
                        ),
                        "\s+",
                        ", ",
                    ),
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
            sorted(atomic, key=lambda x: x["element"], reverse=False)
        )
        row_dict["nperiodic_dimensions"] = json.dumps(
            self.aggregated_info["nperiodic_dimensions"]
        )
        row_dict["dimension_types"] = json.dumps(
            self.aggregated_info["dimension_types"]
        )
        row_dict["potential_energy_count"] = self.aggregated_info[
            "potential_energy_count"
        ]
        row_dict["atomic_forces_count"] = self.aggregated_info["atomic_forces_count"]
        row_dict["cauchy_stress_count"] = self.aggregated_info["cauchy_stress_count"]
        row_dict["free_energy_count"] = self.aggregated_info["free_energy_count"]
        row_dict["authors"] = json.dumps(self.authors)
        row_dict["description"] = self.description
        row_dict["license"] = self.data_license
        row_dict["links"] = json.dumps(self.links)
        row_dict["name"] = self.name

        return (
            self._hash,
            dateutil.parser.parse(
                datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            ),
            len(self.configuration_set_ids),
            self.aggregated_info["nsites"],
            self.aggregated_info["nelements"],
            json.dumps(self.aggregated_info["elements"]),
            json.dumps(self.aggregated_info["total_elements_ratios"]),
            json.dumps(self.aggregated_info["nperiodic_dimensions"]),
            json.dumps(self.aggregated_info["dimension_types"]),
            self.aggregated_info["potential_energy_count"],
            self.aggregated_info["atomic_forces_count"],
            self.aggregated_info["cauchy_stress_count"],
            self.aggregated_info["free_energy_count"],
            json.dumps(self.authors),
            self.description,
            None,
            self.data_license,
            json.dumps(self.links),
            self.name,
        )

    def __hash__(self):
        """Hashes the dataset using its configuration set and property IDs"""
        ds_hash = sha512()

        for i in sorted(self.configuration_set_ids):
            ds_hash.update(str(i).encode("utf-8"))

        for i in sorted(self.property_ids):
            ds_hash.update(str(i).encode("utf-8"))

        return int(ds_hash.hexdigest(), 16)

    def __str__(self):
        return (
            "Dataset(description='{}', nconfiguration_sets={}, nproperties={})".format(
                self.description,
                len(self.configuration_set_ids),
                len(self.property_ids),
            )
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
