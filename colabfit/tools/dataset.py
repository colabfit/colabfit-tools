import datetime
import warnings

import dateutil
import pyspark.sql.functions as sf
from pyspark.sql.types import StringType
from unidecode import unidecode

from colabfit import MAX_STRING_LENGTH
from colabfit.tools.schema import dataset_schema, config_df_schema
from colabfit.tools.utilities import (
    ELEMENT_MAP,
    _empty_dict_from_schema,
    _hash,
    unstring_df_val,
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
        doi: str = None,
        configuration_set_ids: list[str] = [],
        data_license: str = "CC-BY-ND-4.0",
        publication_year: str = None,
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
        self.doi = doi
        self.publication_year = publication_year
        self.configuration_set_ids = configuration_set_ids
        if self.configuration_set_ids is None:
            self.configuration_set_ids = []
        self.spark_row = self.to_spark_row(config_df=config_df, prop_df=prop_df)
        self.spark_row["id"] = self.dataset_id
        # if dataset_id is None:
        #     raise ValueError("Dataset ID must be provided")
        id_prefix = "__".join(
            [
                self.name,
                "-".join([unidecode(auth.split()[-1]) for auth in authors]),
            ]
        )
        if len(id_prefix) > (MAX_STRING_LENGTH - len(dataset_id) - 2):
            id_prefix = id_prefix[: MAX_STRING_LENGTH - len(dataset_id) - 2]
            warnings.warn(f"ID prefix is too long. Clipping to {id_prefix}")
        extended_id = f"{id_prefix}__{dataset_id}"
        self.spark_row["extended_id"] = extended_id
        self._hash = _hash(self.spark_row, ["extended_id"])
        self.spark_row["hash"] = str(self._hash)
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
        config_df = config_df.select(
            "id",
            "elements",
            "atomic_numbers",
            "nsites",
            "nperiodic_dimensions",
            "dimension_types",
            # "labels",
        )
        prop_df = prop_df.select(
            "atomization_energy",
            "atomic_forces_00",
            "adsorption_energy",
            "electronic_band_gap",
            "cauchy_stress",
            "formation_energy",
            "energy",
        )
        carray_cols = ["atomic_numbers", "elements", "dimension_types"]
        carray_types = {
            col.name: col.dataType for col in config_df_schema if col.name in carray_cols
        }
        for col in carray_cols:
            unstr_udf = sf.udf(unstring_df_val, carray_types[col])
            config_df = config_df.withColumn(col, unstr_udf(sf.col(col)))
        row_dict["nsites"] = config_df.agg({"nsites": "sum"}).first()[0]
        row_dict["elements"] = sorted(
            config_df.select("elements")
            .withColumn("exploded_elements", sf.explode("elements"))
            .agg(sf.collect_set("exploded_elements").alias("exploded_elements"))
            .select("exploded_elements")
            .take(1)[0][0]
        )
        row_dict["nelements"] = len(row_dict["elements"])
        atomic_ratios_df = config_df.select("atomic_numbers").withColumn(
            "single_element", sf.explode("atomic_numbers")
        )
        total_elements = atomic_ratios_df.count()
        print(total_elements, row_dict["nsites"])
        assert total_elements == row_dict["nsites"]
        atomic_ratios_df = atomic_ratios_df.groupBy("single_element").count()
        atomic_ratios_df = atomic_ratios_df.withColumn(
            "ratio", sf.col("count") / total_elements
        )
        atomic_ratios_coll = (
            atomic_ratios_df.withColumn(
                "element",
                sf.udf(lambda x: ELEMENT_MAP[x], StringType())(sf.col("single_element")),
            )
            .select("element", "ratio")
            .collect()
        )
        row_dict["total_elements_ratios"] = [
            x[1] for x in sorted(atomic_ratios_coll, key=lambda x: x["element"])
        ]
        row_dict["nperiodic_dimensions"] = config_df.agg(
            sf.collect_set("nperiodic_dimensions")
        ).collect()[0][0]
        row_dict["dimension_types"] = (
            config_df.select("dimension_types")
            .agg(sf.collect_set("dimension_types"))
            .collect()[0][0]
        )
        nproperty_objects = prop_df.count()
        row_dict["nproperty_objects"] = nproperty_objects
        for prop in [
            "atomization_energy",
            "adsorption_energy",
            "electronic_band_gap",
            "cauchy_stress",
            "formation_energy",
            "energy",
        ]:
            row_dict[f"{prop}_count"] = (
                prop_df.select(prop).where(f"{prop} is not null").count()
            )
        row_dict["atomic_forces_count"] = (
            prop_df.select("atomic_forces_00")
            .filter(sf.col("atomic_forces_00") != "[]")
            .count()
        )
        prop = "energy"
        row_dict[f"{prop}_variance"] = (
            prop_df.select(prop).where(f"{prop} is not null").agg(sf.variance(prop))
        ).first()[0]
        row_dict[f"{prop}_mean"] = (
            prop_df.select(prop).where(f"{prop} is not null").agg(sf.mean(prop))
        ).first()[0]
        return row_dict

    def __str__(self):
        return (
            f"Dataset(description='{self.description}', "
            f"nconfiguration_sets={len(self.spark_row['configuration_sets'])}, "
            f"nproperty_objects={self.spark_row['nproperty_objects']}, "
            f"nconfigurations={self.spark_row['nconfigurations']}"
        )

    def __repr__(self):
        return str(self)
