import logging
from datetime import datetime

import pyspark.sql.functions as sf
from unidecode import unidecode

from colabfit import MAX_STRING_LENGTH
from colabfit.tools.vast.schema import dataset_schema
from colabfit.tools.vast.data_object import DataObject
from colabfit.tools.vast.utils import (
    ELEMENT_MAP,
    _empty_dict_from_schema,
    get_date,
    get_last_modified,
)

logger = logging.getLogger(__name__)


class Dataset(DataObject):
    """
    Represents a dataset that aggregates configuration sets and computed properties,
    along with associated metadata and aggregated statistics.

    A Dataset object is used to organize and summarize information about a collection
    of configuration sets and their computed properties, including authorship,
    publication links, licensing, and various aggregated statistics derived from the
    data.

    Parameters
    ----------
    name : str
        The name of the dataset.
    authors : list of str
    publication_link : str
        Link to the publication associated with the dataset.
    data_link : str
        Link to the source data for the dataset.
    description : str
    config_df : pyspark.sql.DataFrame
        DataFrame containing configuration set information.
    other_links : list of str, optional
        Additional external links related to the dataset (default: None).
    dataset_id : str, optional
        Unique identifier for the dataset (default: None).
    labels : list of str, optional
        List of labels associated with the dataset (default: None).
    doi : str, optional
        Digital Object Identifier for the dataset (default: None).
    configuration_set_ids : list of str, optional
        List of configuration set IDs attached to the dataset (default: []).
    data_license : str, optional
        License associated with the dataset's data (default: "CC-BY-ND-4.0").
    publication_year : str, optional
        Year of publication for the dataset (default: None).

    Attributes
    ----------
    name : str
        The name of the dataset.
    authors : list of str
    publication_link : str
        Link to the publication associated with the dataset.
    data_link : str
        Link to the source data for the dataset.
    other_links : list of str or None
        Additional external links related to the dataset.
    description : str
    data_license : str
        License associated with the dataset's data.
    dataset_id : str or None
        Unique identifier for the dataset.
    doi : str or None
        Digital Object Identifier for the dataset.
    publication_year : str or None
        Year of publication for the dataset.
    configuration_set_ids : list of str
        List of configuration set IDs attached to the dataset.
    row_dict : dict
        Dictionary containing aggregated statistics and metadata for the dataset,
        appropriate for use in a Spark DataFrame, Vast DB table or similar.
        Includes:
            - nconfigurations: Number of configurations.
            - nsites: Total number of sites.
            - nelements: Number of unique elements.
            - elements: List of unique elements.
            - nperiodic_dimensions: Distinct periodic dimensions.
            - dimension_types: Distinct dimension types.
            - total_elements_ratios: Ratios of each element in the dataset.
            - nproperty_objects: Number of property objects.
            - Various property counts and statistics.
            - dataset authors
            - dataset description
            - dataset license
            - links to publication, data, and resources associated with the dataset.
            - dataset name
            - dataset publication_year
            - doi: Digital Object Identifier for the dataset.
    labels : list of str or None
        List of labels associated with the dataset.

    Methods
    -------
    to_row_dict(config_df)
        Aggregates statistics and metadata from the configuration DataFrame
        into a dictionary representation suitable for use in a Spark DataFrame, Vast DB
        table or similar.

    __str__()
        Returns a string summary of the dataset.

    __repr__()
        Returns a string representation of the dataset.
    """

    def __init__(
        self,
        name: str,
        authors: list[str],
        publication_link: str,
        data_link: str,
        description: str,
        config_df,
        other_links: list[str] = None,
        dataset_id: str = None,
        labels: list[str] = None,
        doi: str = None,
        configuration_set_ids: list[str] = [],
        data_license: str = None,
        date_requested: str = None,
        publication_year: str = None,
        equilibrium: bool = False,
    ):
        for auth in authors:
            if not "".join(auth.split(" ")[-1].replace("-", "")).isalpha():
                raise RuntimeError(
                    f"Bad author name '{auth}'. Author names "
                    "can only contain [a-z][A-Z]"
                )
        for required in (date_requested, data_license, publication_year, dataset_id):
            if not required:
                raise RuntimeError(f"Missing required field {required}")
        self.name = name
        self.authors = authors
        self.publication_link = publication_link
        self.data_link = data_link
        self.other_links = other_links
        self.description = description
        self.data_license = data_license
        self.dataset_id = dataset_id
        self.doi = doi
        assert datetime.strptime(publication_year, "%Y")
        self.publication_year = publication_year
        self.configuration_set_ids = configuration_set_ids
        self.equilibrium = equilibrium
        if self.configuration_set_ids is None:
            self.configuration_set_ids = []
        self.row_dict = self.to_row_dict(config_df=config_df)
        self.row_dict["date_added_to_colabfit"] = get_date()
        assert datetime.strptime(date_requested, "%Y-%m-%d")
        self.row_dict["date_requested"] = datetime.strptime(date_requested, "%Y-%m-%d")
        id_prefix = "__".join(
            [
                self.name,
                "-".join([unidecode(auth.split()[-1]) for auth in authors]),
            ]
        )
        if len(id_prefix) > (MAX_STRING_LENGTH - len(dataset_id) - 2):
            id_prefix = id_prefix[: MAX_STRING_LENGTH - len(dataset_id) - 2]
            logger.warning(f"ID prefix is too long. Clipping to {id_prefix}")
        extended_id = f"{id_prefix}__{dataset_id}"
        self.row_dict["extended_id"] = extended_id
        self._generate_hash_and_id()
        self.row_dict["id"] = self.dataset_id
        self.row_dict["labels"] = labels
        logger.info(self.row_dict)

    def get_identifier_keys(self) -> list[str]:
        """Return the keys used for Dataset identification."""
        return ["extended_id"]

    def to_row_dict(self, config_df):
        """"""
        row_dict = _empty_dict_from_schema(dataset_schema)
        row_dict["last_modified"] = get_last_modified()
        row_dict["nconfiguration_sets"] = len(self.configuration_set_ids)
        config_df = config_df.select(
            "property_id",
            "configuration_id",
            "elements",
            "atomic_numbers",
            "nsites",
            "nperiodic_dimensions",
            "dimension_types",
            "atomization_energy",
            "atomic_forces",
            "adsorption_energy",
            "electronic_band_gap",
            "energy_above_hull",
            "cauchy_stress",
            "formation_energy",
            "energy",
            "method",
            "software",
        )
        config_df.persist()
        row_dict["nconfigurations"] = (
            config_df.select("configuration_id").distinct().count()
        )
        row_dict["nsites"] = (
            config_df.select("nsites").agg(sf.sum("nsites")).collect()[0][0]
        )
        nperiodic_dims = config_df.select("nperiodic_dimensions").distinct().collect()
        row_dict["nperiodic_dimensions"] = [
            row["nperiodic_dimensions"] for row in nperiodic_dims
        ]
        dim_types = config_df.select("dimension_types").distinct().collect()
        row_dict["dimension_types"] = [row["dimension_types"] for row in dim_types]

        methods = config_df.select("method").distinct().collect()
        row_dict["methods"] = [row["method"] for row in methods]

        software = config_df.select("software").distinct().collect()
        row_dict["software"] = [row["software"] for row in software]

        elements_df = config_df.select("elements").distinct()
        elements = []
        for row in elements_df.collect():
            elem_list = (
                row["elements"]
                if isinstance(row["elements"], list)
                else [row["elements"]]
            )
            elements.extend(elem_list)
        row_dict["elements"] = sorted(list(set(elements)))
        row_dict["nelements"] = len(row_dict["elements"])
        atomic_ratios_df = (
            config_df.select("atomic_numbers")
            .withColumn("single_element", sf.explode("atomic_numbers"))
            .groupBy("single_element")
            .agg(sf.count("single_element").alias("count"))
        )
        total_elements = atomic_ratios_df.agg(sf.sum("count")).collect()[0][0]
        atomic_ratios_df = atomic_ratios_df.withColumn(
            "ratio", sf.col("count") / total_elements
        )
        logger.info(f'{total_elements} {row_dict["nsites"]}')
        assert total_elements == row_dict["nsites"]

        element_map_expr = sf.create_map(
            [
                sf.lit(k)
                for pair in [(k, v) for k, v in ELEMENT_MAP.items()]
                for k in pair
            ]
        )

        atomic_ratios_coll = (
            atomic_ratios_df.withColumn(
                "element", element_map_expr[sf.col("single_element")]
            )
            .select("element", "ratio")
            .collect()
        )
        row_dict["total_elements_ratios"] = [
            x["ratio"] for x in sorted(atomic_ratios_coll, key=lambda x: x["element"])
        ]
        del atomic_ratios_df, atomic_ratios_coll

        row_dict["nproperty_objects"] = (
            config_df.select("property_id").distinct().count()
        )
        property_counts = {}
        property_cols = [
            "atomization_energy",
            "adsorption_energy",
            "electronic_band_gap",
            "cauchy_stress",
            "energy_above_hull",
            "formation_energy",
            "energy",
        ]
        for prop_col in property_cols:
            count = (
                config_df.select(prop_col).filter(sf.col(prop_col).isNotNull()).count()
            )
            property_counts[f"{prop_col}_count"] = count
        atomic_forces_count = config_df.filter(
            (sf.col("atomic_forces") != "[]") & (sf.col("atomic_forces").isNotNull())
        ).count()
        property_counts["atomic_forces_count"] = atomic_forces_count
        energy_stats = (
            config_df.select("energy")
            .filter(sf.col("energy").isNotNull())
            .agg(
                sf.variance("energy").alias("energy_variance"),
                sf.mean("energy").alias("energy_mean"),
            )
            .collect()[0]
        )
        property_counts["energy_variance"] = energy_stats["energy_variance"]
        property_counts["energy_mean"] = energy_stats["energy_mean"]
        row_dict.update(property_counts)
        row_dict["authors"] = self.authors
        row_dict["description"] = self.description
        row_dict["license"] = self.data_license
        row_dict["links"] = str(
            {
                "source-publication": self.publication_link,
                "source-data": self.data_link,
                "other": self.other_links,
            }
        )
        row_dict["name"] = self.name
        row_dict["publication_year"] = self.publication_year
        row_dict["doi"] = self.doi
        row_dict["equilibrium"] = self.equilibrium
        config_df.unpersist()
        del config_df
        return row_dict

    def __str__(self):
        return (
            f"Dataset(description='{self.description}', "
            f"nconfiguration_sets={len(self.row_dict['configuration_sets'])}, "
            f"nproperty_objects={self.row_dict['nproperty_objects']}, "
            f"nconfigurations={self.row_dict['nconfigurations']}"
        )

    def __repr__(self):
        return str(self)
