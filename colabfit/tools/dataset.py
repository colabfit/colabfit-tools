from hashlib import sha512


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
            data_license='CC-BY-ND-4.0'
    ):

        for auth in authors:
            if not ''.join(auth.split(' ')[-1].replace('-', '')).isalpha():
                raise RuntimeError(
                    "Bad author name '{}'. Author names can only contain [a-z][A-Z]".format(auth)
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

    def __hash__(self):
        """Hashes the dataset using its configuration set and property IDs"""
        ds_hash = sha512()

        for i in sorted(self.configuration_set_ids):
            ds_hash.update(str(i).encode('utf-8'))

        for i in sorted(self.property_ids):
            ds_hash.update(str(i).encode('utf-8'))

        return int(ds_hash.hexdigest(), 16)

    def __str__(self):
        return "Dataset(description='{}', nconfiguration_sets={}, nproperties={})".format(
            self.description,
            len(self.configuration_set_ids),
            len(self.property_ids),
        )

    def __repr__(self):
        return str(self)
