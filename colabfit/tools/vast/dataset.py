import itertools
import logging
from collections import Counter
from datetime import datetime

import pyarrow as pa
import pyarrow.compute as pc
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
    config_df : pa.Table
        Table containing configuration/property data.
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

    def to_row_dict(self, config_df: pa.Table):
        row_dict = _empty_dict_from_schema(dataset_schema)
        row_dict["last_modified"] = get_last_modified()
        row_dict["nconfiguration_sets"] = len(self.configuration_set_ids)

        config_df = config_df.select(
            [
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
            ]
        )

        row_dict["nconfigurations"] = len(
            pc.unique(config_df["configuration_id"].drop_null())
        )
        row_dict["nsites"] = pc.sum(config_df["nsites"]).as_py()

        row_dict["nperiodic_dimensions"] = (
            pc.unique(config_df["nperiodic_dimensions"]).drop_null().to_pylist()
        )

        seen_dim_types = set()
        unique_dim_types = []
        for dt in config_df["dimension_types"].to_pylist():
            if dt is not None:
                key = tuple(dt)
                if key not in seen_dim_types:
                    seen_dim_types.add(key)
                    unique_dim_types.append(dt)
        row_dict["dimension_types"] = unique_dim_types

        row_dict["methods"] = pc.unique(config_df["method"]).drop_null().to_pylist()
        row_dict["software"] = pc.unique(config_df["software"]).drop_null().to_pylist()

        all_elements = set()
        for elem_list in config_df["elements"].to_pylist():
            if elem_list:
                all_elements.update(elem_list)
        row_dict["elements"] = sorted(list(all_elements))
        row_dict["nelements"] = len(row_dict["elements"])

        all_nums = list(
            itertools.chain.from_iterable(
                row for row in config_df["atomic_numbers"].to_pylist() if row
            )
        )
        element_counts = Counter(all_nums)
        total_elements = sum(element_counts.values())
        logger.info(f'{total_elements} {row_dict["nsites"]}')
        assert total_elements == row_dict["nsites"]

        element_symbol_counts = {
            ELEMENT_MAP[num]: count for num, count in element_counts.items()
        }
        sorted_symbols = sorted(element_symbol_counts.keys())
        row_dict["total_elements_ratios"] = [
            element_symbol_counts[sym] / total_elements for sym in sorted_symbols
        ]

        row_dict["nproperty_objects"] = len(
            pc.unique(config_df["property_id"].drop_null())
        )

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
            count = pc.count(config_df[prop_col], count_mode="only_valid").as_py()
            row_dict[f"{prop_col}_count"] = count

        valid_forces = pc.and_(
            pc.is_valid(config_df["atomic_forces"]),
            pc.greater(pc.list_size(config_df["atomic_forces"]), 0),
        )
        row_dict["atomic_forces_count"] = pc.sum(valid_forces.cast(pa.int64())).as_py()

        energy_valid = config_df.filter(pc.is_valid(config_df["energy"]))["energy"]
        if len(energy_valid) > 0:
            row_dict["energy_variance"] = pc.variance(energy_valid).as_py()
            row_dict["energy_mean"] = pc.mean(energy_valid).as_py()
        else:
            row_dict["energy_variance"] = None
            row_dict["energy_mean"] = None

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
