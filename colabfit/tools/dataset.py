from hashlib import sha512

from colabfit import HASH_SHIFT

class Dataset:
    """
    A dataset defines a group of configuration sets and computed properties, and
    aggregates information about those configuration sets and properties.

    Attributes:

        configuration_set_ids (list):
            A list of attached configuration sets

        property_ids (list):
            A list of attached properties

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

                * nconfigurations: the total number of configurations
                * nsites: the total number of sites
                * nelements: the total number of unique element types
                * elements: the element types
                * individual_elements_ratios: a set of elements ratios generated
                by looping over each configuration, extracting its concentration of
                each element, and adding the tuple of concentrations to the set
                * total_elements_ratios: the ratio of the total count of atoms
                    of each element type over nsites
                * configuration_labels: the union of all configuration labels
                * configuration_labels_counts: the total count of each configuration label
                * chemical_formula_reduced: the set of all reduced chemical
                    formulae
                * chemical_formula_anonymous: the set of all anonymous chemical
                    formulae
                * chemical_formula_hill: the set of all hill chemical formulae
                * nperiodic_dimensions: the set of all numbers of periodic
                    dimensions
                * dimension_types: the set of all periodic boundary choices

                From the properties:

                * types: the set of all property types
                * property_labels: the set of all property labels
                * property_labels_counts: the total count of each property label
    """

    def __init__(
        self,
        configuration_set_ids,
        property_ids,
        authors,
        links,
        description,
        aggregated_info,
        ):

        self.configuration_set_ids  = configuration_set_ids
        self.property_ids           = property_ids
        self.authors                = authors
        self.links                  = links
        self.description            = description
        self.aggregated_info        = aggregated_info


    def __hash__(self):
        ds_hash = sha512()

        for i in sorted(self.configuration_set_ids):
            ds_hash.update(str(i).encode('utf-8'))

        for i in sorted(self.property_ids):
            ds_hash.update(str(i).encode('utf-8'))

        return int(ds_hash.hexdigest()[:16], 16)-HASH_SHIFT