from hashlib import sha512
from pyspark.sql import functions as sf
from colabfit.tools.utilities import _empty_dict_from_schema
from colabfit.tools.schema import configuration_set_schema
from pyspark.sql.types import IntegerType, StringType
from datetime import datetime
import dateutil.parser

from colabfit.tools.utilities import ELEMENT_MAP


class ConfigurationSet:
    """
    A configuration set defines a group of configurations and aggregates
    information about those configurations to improve queries.

    Note that a configuration set should only be constructed by loading from an
    existing database (in order to aggregate the info)

    Attributes:

        configuration_ids (list):
            A list of all attached configuration IDs

        description (str):
            A human-readable description of the configuration set

        aggregated_info (dict):
            A dictionary of information that was aggregated from all of the
            attached configurations. Contains the following information:

                * nconfigurations: the total number of configurations
                * nsites: the total number of sites
                * nelements: the total number of unique element types
                * elements: the element types
                * individual_elements_ratios: a set of elements ratios generated by
                  looping over each configuration, extracting its concentration of each
                  element, and adding the tuple of concentrations to the set
                * total_elements_ratios: the ratio of the total count of atoms of each
                  element type over nsites
                * chemical_formula_reduced: the set of all reduced chemical formulae
                * chemical_formula_anonymous: the set of all anonymous chemical formulae
                * chemical_formula_hill: the set of all hill chemical formulae
                * nperiodic_dimensions: the set of all numbers of periodic dimensions
                * dimension_types: the set of all periodic boundary choices

    """

    def __init__(self, config_df, name, description, dataset_id, ordered=False):
        self.name = name
        self.description = description
        self.dataset_id = dataset_id
        # self.ordered = ordered
        self.spark_row = self.to_spark_row(config_df)
        self._hash = hash(self)
        self.id = f"CS_{self._hash}"
        self.spark_row["id"] = self.id
        self.spark_row["hash"] = self._hash

    def to_spark_row(self, config_df):
        row_dict = _empty_dict_from_schema(configuration_set_schema)
        row_dict["name"] = self.name
        row_dict["description"] = self.description
        row_dict["nconfigurations"] = config_df.count()
        row_dict["last_modified"] = dateutil.parser.parse(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        row_dict["nsites"] = config_df.agg({"nsites": "sum"}).first()[0]
        row_dict["elements"] = sorted(
            config_df.withColumn("exploded_elements", sf.explode("elements"))
            .agg(sf.collect_set("exploded_elements").alias("exploded_elements"))
            .select("exploded_elements")
            .take(1)[0][0]
        )
        atomic_ratios_df = (
            config_df.select("atomic_numbers")
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
        row_dict["nelements"] = len(row_dict["elements"])
        row_dict["nsites"] = config_df.agg({"nsites": "sum"}).first()[0]
        row_dict["dataset_id"] = self.dataset_id
        return row_dict

    def __hash__(self):
        cs_hash = sha512()
        cs_hash.update(self.name.encode("utf-8"))
        return int(cs_hash.hexdigest(), 16)

    def __str__(self):
        return "ConfigurationSet(description='{}', nconfigurations={})".format(
            self.description,
            self.spark_row["nconfigurations"],
        )

    def __repr__(self):
        return str(self)
