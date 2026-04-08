"""
"""

from collections import namedtuple

column = namedtuple("field", ["name", "type", "nullable"])


class Schema:
    def __init__(self, name: str, columns: list[tuple]):
        self.name = name
        self.columns = columns
        self.column_names = [column.name for column in columns]

    def add(self, column: tuple):
        return Schema(self.name, self.columns + [column])

    def __str__(self):
        return str(self.columns)

    def __repr__(self):
        return str(self.columns)

    def __eq__(self, other):
        return self.name == other.name and self.columns == other.columns

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.columns + [self.name])


config_schema = Schema(
    name="configurations",
    columns=[
        column("id", "VARCHAR (256)", False),
        column("hash", "VARCHAR (256) PRIMARY KEY", True),
        column("last_modified", "VARCHAR (256)", True),
        column("dataset_ids", "VARCHAR (256) []", True),
        column("configuration_set_ids", "VARCHAR (256) []", True),
        column("chemical_formula_hill", "VARCHAR (256)", True),
        column("chemical_formula_reduced", "VARCHAR (256)", True),
        column("chemical_formula_anonymous", "VARCHAR (256)", True),
        column("elements", "VARCHAR (256) []", True),
        column("elements_ratios", "DOUBLE PRECISION []", True),
        column("atomic_numbers", "INT []", True),
        column("nsites", "INT", True),
        column("nelements", "INT", True),
        column("nperiodic_dimensions", "INT", True),
        column("cell", "DOUBLE PRECISION [] []", True),
        column("dimension_types", "INT []", True),
        column("pbc", "BOOL []", True),
        column("names", "VARCHAR (256) []", True),
        column("labels", "VARCHAR (256) []", True),
        column("positions", "DOUBLE PRECISION [] []", True),
    ],
)

config_md_schema = config_schema.add(column("metadata", "VARCHAR (10000)", True))


property_object_schema = Schema(
    name="property_objects",
    columns=[
        column("id", "VARCHAR (256)", False),
        column("hash", "VARCHAR (256) PRIMARY KEY", True),
        column("last_modified", "VARCHAR (256)", True),
        column("configuration_id", "VARCHAR (256)", True),
        column("dataset_id", "VARCHAR (256)", True),
        column("multiplicity", "INT", True),
        column("software", "VARCHAR (256)", True),
        column("method", "VARCHAR (256)", True),
        column("chemical_formula_hill", "VARCHAR (256)", True),
        column("energy", "DOUBLE PRECISION", True),
        column("atomic_forces", "DOUBLE PRECISION [] []", True),
        column("cauchy_stress", "DOUBLE PRECISION [] []", True),
        column("cauchy_stress_volume_normalized", "BOOL", True),
        column("electronic_band_gap", "DOUBLE PRECISION", True),
        column("electronic_band_gap_type", "VARCHAR (256)", True),
        column("formation_energy", "DOUBLE PRECISION", True),
        column("adsorption_energy", "DOUBLE PRECISION", True),
        column("atomization_energy", "DOUBLE PRECISION", True),
    ],
)

property_object_md_schema = property_object_schema.add(
    column("metadata", "VARCHAR (10000)", True)
)

dataset_schema = Schema(
    name="datasets",
    columns=[
        column("id", "VARCHAR (256)", False),
        column("hash", "VARCHAR (256) PRIMARY KEY", False),
        column("name", "VARCHAR (256)", True),
        column("last_modified", "VARCHAR (256)", True),
        column("nconfigurations", "INT", True),
        column("nproperty_objects", "INT", True),
        column("nsites", "INT", True),
        column("nelements", "INT", True),
        column("elements", "VARCHAR (1000) []", True),
        column("total_elements_ratios", "DOUBLE PRECISION []", True),
        column("nperiodic_dimensions", "INT []", True),
        column("dimension_types", "VARCHAR (1000) []", True),
        column("energy_count", "INT", True),
        column("energy_mean", "DOUBLE PRECISION", True),
        column("energy_variance", "DOUBLE PRECISION", True),
        column("atomization_energy_count", "INT", True),
        column("adsorption_energy_count", "INT", True),
        column("formation_energy_count", "INT", True),
        column("atomic_forces_count", "INT", True),
        column("electronic_band_gap_count", "INT", True),
        column("cauchy_stress_count", "INT", True),
        column("authors", "VARCHAR (256)", True),
        column("description", "VARCHAR (10000)", True),
        column("extended_id", "VARCHAR (1000)", True),
        column("license", "VARCHAR (256)", True),
        column("links", "VARCHAR (1000) []", True),
        column("publication_year", "VARCHAR (256)", True),
        column("doi", "VARCHAR (256)", True),
    ],
)


configuration_set_schema = Schema(
    name="configuration_sets",
    columns=[
        column("id", "VARCHAR (256)", False),
        column("hash", "VARCHAR (256) PRIMARY KEY", False),
        column("last_modified", "VARCHAR (256)", True),
        column("nconfigurations", "INT", True),
        column("nperiodic_dimensions", "INT []", True),
        column("dimension_types", "INT []", True),
        column("nsites", "INT", True),
        column("nelements", "INT", True),
        column("elements", "VARCHAR (1000) []", True),
        column("total_elements_ratios", "DOUBLE PRECISION []", True),
        column("description", "VARCHAR (10000)", True),
        column("name", "VARCHAR (256)", True),
        column("dataset_id", "VARCHAR (256)", True),
        column("ordered", "BOOL", True),
        column("extended_id", "VARCHAR (256)", True),
    ],
)

co_cs_mapping_schema = Schema(
    name="config_set_mapping",
    columns=[
        column("configuration_id", "VARCHAR (256)", True),
        column("configuration_set_id", "VARCHAR (256)", True),
    ],
)

property_definition_schema = Schema(
    name="property_definitions",
    columns=[
        column("hash", "VARCHAR (256) PRIMARY KEY", True),
        column("last_modified", "VARCHAR (256)", True),
        column("definition", "VARCHAR (10000)", True),
    ],
)
