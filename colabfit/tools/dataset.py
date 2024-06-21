import datetime
import warnings

import dateutil
import numpy as np
import pyspark.sql.functions as sf
from unidecode import unidecode
from pyspark.sql.types import StringType


from colabfit import MAX_STRING_LENGTH
from colabfit.tools.schema import dataset_schema
from colabfit.tools.utilities import (
    _empty_dict_from_schema,
    ELEMENT_MAP,
)


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
        config_df,
        prop_df,
        other_links: list[str] = None,
        dataset_id: str = None,
        labels: list[str] = None,
        configuration_set_ids: list[str] = [],
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
        self.dataset_id = dataset_id
        self.configuration_set_ids = configuration_set_ids
        self.spark_row = self.to_spark_row(config_df=config_df, prop_df=prop_df)
        self._hash = hash(self)
        self.spark_row["hash"] = self._hash
        self.spark_row["id"] = self.dataset_id
        if dataset_id is None:
            raise ValueError("Dataset ID must be provided")
        id_prefix = "_".join(
            [
                self.name,
                "".join([unidecode(auth.split()[-1]) for auth in authors]),
            ]
        )
        if len(id_prefix) > (MAX_STRING_LENGTH - len(dataset_id) - 2):
            id_prefix = id_prefix[: MAX_STRING_LENGTH - len(dataset_id) - 2]
            warnings.warn(f"ID prefix is too long. Clipping to {id_prefix}")
        extended_id = f"{id_prefix}__{dataset_id}"
        self.spark_row["extended_id"] = extended_id
        self.spark_row["labels"] = labels
        print(self.spark_row)

    def to_spark_row(self, config_df, prop_df):
        """"""
        row_dict = _empty_dict_from_schema(dataset_schema)
        row_dict["last_modified"] = dateutil.parser.parse(
            datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        )
        row_dict["nconfiguration_sets"] = len(self.configuration_set_ids)
        config_df = config_df.withColumnRenamed("id", "config_id").withColumnRenamed(
            "hash", "config_hash"
        )
        co_po_df = prop_df.join(
            config_df, prop_df["configuration_id"] == config_df["config_id"], "inner"
        )
        row_dict["nsites"] = co_po_df.agg({"nsites": "sum"}).first()[0]
        row_dict["elements"] = sorted(
            co_po_df.withColumn("exploded_elements", sf.explode("elements"))
            .agg(sf.collect_set("exploded_elements").alias("exploded_elements"))
            .select("exploded_elements")
            .take(1)[0][0]
        )
        row_dict["nelements"] = len(row_dict["elements"])
        atomic_ratios_df = (
            co_po_df.select("atomic_numbers")
            .withColumn("exploded_atom", sf.explode("atomic_numbers"))
            .groupBy(sf.col("exploded_atom").alias("atomic_number"))
            .count()
            .withColumn("ratio", sf.col("count") / row_dict["nsites"])
            .select("ratio", "atomic_number")
            .withColumn(
                "element",
                sf.udf(lambda x: ELEMENT_MAP[x], StringType())(sf.col("atomic_number")),
            )
            .select("element", "ratio")
            .collect()
        )
        row_dict["total_elements_ratios"] = [
            x[1] for x in sorted(atomic_ratios_df, key=lambda x: x["element"])
        ]

        row_dict["nperiodic_dimensions"] = co_po_df.agg(
            sf.collect_set("nperiodic_dimensions")
        ).collect()[0][0]

        row_dict["dimension_types"] = (
            co_po_df.select("dimension_types")
            .agg(sf.collect_set("dimension_types"))
            .collect()[0][0]
        )

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
        row_dict["nconfigurations"] = co_po_df.count()
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
        return hash(self.name)

    def __str__(self):
        return (
            f"Dataset(description='{self.description}', "
            f"nconfiguration_sets={len(self.spark_row['configuration_sets'])}, "
            f"nproperties={self.spark_row['nproperties']})"
            f"nconfigurations={self.spark_row['nconfigurations']}"
        )

    def __repr__(self):
        return str(self)
