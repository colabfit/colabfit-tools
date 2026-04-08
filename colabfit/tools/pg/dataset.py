import warnings
from unidecode import unidecode

from colabfit import MAX_STRING_LENGTH
from colabfit.tools.pg.utilities import ELEMENT_MAP, _hash, get_last_modified

import numpy as np


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
        self.row_dict = self.to_row_dict(configs=config_df, props=prop_df)
        self.row_dict["id"] = self.dataset_id

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
        self.row_dict["extended_id"] = extended_id
        self._hash = _hash(self.row_dict, ["extended_id"])
        self.row_dict["hash"] = str(self._hash)
        self.row_dict["labels"] = labels

    # aggregate stuff
    def to_row_dict(self, configs, props):
        row_dict = {}
        row_dict["last_modified"] = get_last_modified()
        row_dict["nconfiguration_sets"] = len(self.configuration_set_ids)
        row_dict["nconfigurations"] = len(configs)
        row_dict["nproperty_objects"] = len(props)
        nsites = 0
        nperiodic_dimensions = set()
        dimension_types = set()
        element_dict = {}

        for c in configs:
            nsites += c["nsites"]
            for e in c["atomic_numbers"]:
                e = ELEMENT_MAP[e]
                if e in element_dict:
                    element_dict[e] += 1
                else:
                    element_dict[e] = 1
            nperiodic_dimensions.add(c["nperiodic_dimensions"])
            dimension_types.add(str(c["dimension_types"]))

        sorted_elements = sorted(list(element_dict.keys()))

        row_dict["nsites"] = nsites
        row_dict["nelements"] = len(sorted_elements)
        row_dict["elements"] = sorted_elements
        row_dict["total_elements_ratio"] = [
            element_dict[e] / nsites for e in sorted_elements
        ]
        row_dict["nperiodic_dimensions"] = list(nperiodic_dimensions)
        row_dict["dimension_types"] = list(dimension_types)

        forces = 0
        stress = 0
        energy = 0
        energies = []
        for p in props:
            if p["atomic_forces_forces"] is not None:
                forces += 1
            if p["cauchy_stress_stress"] is not None:
                stress += 1
            if p["energy_energy"] is not None:
                energy += 1
                energies.append(p["energy_energy"])

        row_dict["energy_mean"] = np.mean(energies)
        row_dict["energy_variance"] = np.var(energies)
        row_dict["atomic_forces_count"] = forces
        row_dict["cauchy_stress_count"] = stress
        row_dict["energy_count"] = energy
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
