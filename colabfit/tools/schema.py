"""
Each schema for VastDB has a corresponding schema for a dataframe with
the same fields, but with non-stringified lists.
config_schema (with stringified lists), for example, has a corresponding
config_df_schema (with non-stringified lists).
"""

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

config_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", StringType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("dataset_ids", StringType(), True),  # ArrayType(StringType())
        StructField("metadata", StringType(), True),
        StructField("chemical_formula_hill", StringType(), True),
        StructField("chemical_formula_reduced", StringType(), True),
        StructField("chemical_formula_anonymous", StringType(), True),
        StructField("elements", StringType(), True),  # ArrayType(StringType())
        StructField("elements_ratios", StringType(), True),  # ArrayType(IntegerType())
        StructField("atomic_numbers", StringType(), True),  # ArrayType(IntegerType())
        StructField("nsites", IntegerType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("nperiodic_dimensions", IntegerType(), True),
        StructField("cell", StringType(), True),  # ArrayType(ArrayType(DoubleType()))
        StructField("dimension_types", StringType(), True),  # ArrayType(IntegerType())
        StructField("pbc", StringType(), True),  # ArrayType(IntegerType())
        StructField(
            "positions", StringType(), True
        ),  # ArrayType(ArrayType(DoubleType()))
        StructField("names", StringType(), True),  # ArrayType(StringType()),
        StructField("labels", StringType(), True),  # ArrayType(StringType())
        StructField(
            "configuration_set_ids", StringType(), True
        ),  # ArrayType(StringType())
    ]
)

config_df_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", StringType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("dataset_ids", ArrayType(StringType()), True),
        StructField("metadata", StringType(), True),
        StructField("chemical_formula_hill", StringType(), True),
        StructField("chemical_formula_reduced", StringType(), True),
        StructField("chemical_formula_anonymous", StringType(), True),
        StructField("elements", ArrayType(StringType()), True),
        StructField("elements_ratios", ArrayType(DoubleType()), True),
        StructField("atomic_numbers", ArrayType(IntegerType()), True),
        StructField("nsites", IntegerType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("nperiodic_dimensions", IntegerType(), True),
        StructField("cell", ArrayType(ArrayType(DoubleType())), True),
        StructField("dimension_types", ArrayType(IntegerType()), True),
        StructField("pbc", ArrayType(BooleanType()), True),
        StructField("positions", ArrayType(ArrayType(DoubleType())), True),
        StructField("names", ArrayType(StringType()), True),
        StructField("labels", ArrayType(StringType()), True),
        StructField("configuration_set_ids", ArrayType(StringType()), True),
    ]
)


property_object_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", StringType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("configuration_ids", StringType(), True),  # ArrayType(StringType())
        StructField("dataset_ids", StringType(), True),  # ArrayType(StringType())
        StructField("metadata", StringType(), True),
        StructField("software", StringType(), True),
        StructField("method", StringType(), True),
        StructField("chemical_formula_hill", StringType(), True),
        StructField("potential_energy", DoubleType(), True),
        StructField("potential_energy_unit", StringType(), True),
        StructField("potential_energy_per_atom", BooleanType(), True),
        StructField("potential_energy_reference", DoubleType(), True),
        StructField("potential_energy_reference_unit", StringType(), True),
        StructField("potential_energy_property_id", StringType(), True),
        StructField(
            "atomic_forces", StringType(), True
        ),  # ArrayType(ArrayType(DoubleType()))
        StructField("atomic_forces_unit", StringType(), True),
        StructField("atomic_forces_property_id", StringType(), True),
        StructField(
            "cauchy_stress", StringType(), True
        ),  # ArrayType(ArrayType(DoubleType()))
        StructField("cauchy_stress_unit", StringType(), True),
        StructField("cauchy_stress_volume_normalized", BooleanType(), True),
        StructField("cauchy_stress_property_id", StringType(), True),
        StructField("free_energy", DoubleType(), True),
        StructField("free_energy_unit", StringType(), True),
        StructField("free_energy_per_atom", BooleanType(), True),
        StructField("free_energy_reference", DoubleType(), True),
        StructField("free_energy_reference_unit", StringType(), True),
        StructField("free_energy_property_id", StringType(), True),
        StructField("band_gap", DoubleType(), True),
        StructField("band_gap_unit", StringType(), True),
        StructField("band_gap_property_id", StringType(), True),
        StructField("formation_energy", DoubleType(), True),
        StructField("formation_energy_unit", StringType(), True),
        StructField("formation_energy_per_atom", BooleanType(), True),
        StructField("formation_energy_reference", DoubleType(), True),
        StructField("formation_energy_reference_unit", StringType(), True),
        StructField("formation_energy_property_id", StringType(), True),
        StructField("adsorption_energy", DoubleType(), True),
        StructField("adsorption_energy_unit", StringType(), True),
        StructField("adsorption_energy_per_atom", BooleanType(), True),
        StructField("adsorption_energy_reference", DoubleType(), True),
        StructField("adsorption_energy_reference_unit", StringType(), True),
        StructField("adsorption_energy_property_id", StringType(), True),
        StructField("atomization_energy", DoubleType(), True),
        StructField("atomization_energy_unit", StringType(), True),
        StructField("atomization_energy_per_atom", BooleanType(), True),
        StructField("atomization_energy_reference", DoubleType(), True),
        StructField("atomization_energy_reference_unit", StringType(), True),
        StructField("atomization_energy_property_id", StringType(), True),
    ]
)

property_object_df_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", StringType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("configuration_ids", ArrayType(StringType()), True),
        StructField("dataset_ids", ArrayType(StringType()), True),
        StructField("metadata", StringType(), True),
        StructField("software", StringType(), True),
        StructField("method", StringType(), True),
        StructField("chemical_formula_hill", StringType(), True),
        StructField("potential_energy", DoubleType(), True),
        StructField("potential_energy_unit", StringType(), True),
        StructField("potential_energy_per_atom", BooleanType(), True),
        StructField("potential_energy_reference", DoubleType(), True),
        StructField("potential_energy_reference_unit", StringType(), True),
        StructField("potential_energy_property_id", StringType(), True),
        StructField("atomic_forces", ArrayType(ArrayType(DoubleType())), True),
        StructField("atomic_forces_unit", StringType(), True),
        StructField("atomic_forces_property_id", StringType(), True),
        StructField("cauchy_stress", ArrayType(ArrayType(DoubleType())), True),
        StructField("cauchy_stress_unit", StringType(), True),
        StructField("cauchy_stress_volume_normalized", BooleanType(), True),
        StructField("cauchy_stress_property_id", StringType(), True),
        StructField("free_energy", DoubleType(), True),
        StructField("free_energy_unit", StringType(), True),
        StructField("free_energy_per_atom", BooleanType(), True),
        StructField("free_energy_reference", DoubleType(), True),
        StructField("free_energy_reference_unit", StringType(), True),
        StructField("free_energy_property_id", StringType(), True),
        StructField("band_gap", DoubleType(), True),
        StructField("band_gap_unit", StringType(), True),
        StructField("band_gap_property_id", StringType(), True),
        StructField("formation_energy", DoubleType(), True),
        StructField("formation_energy_unit", StringType(), True),
        StructField("formation_energy_per_atom", BooleanType(), True),
        StructField("formation_energy_reference", DoubleType(), True),
        StructField("formation_energy_reference_unit", StringType(), True),
        StructField("formation_energy_property_id", StringType(), True),
        StructField("adsorption_energy", DoubleType(), True),
        StructField("adsorption_energy_unit", StringType(), True),
        StructField("adsorption_energy_per_atom", BooleanType(), True),
        StructField("adsorption_energy_reference", DoubleType(), True),
        StructField("adsorption_energy_reference_unit", StringType(), True),
        StructField("adsorption_energy_property_id", StringType(), True),
        StructField("atomization_energy", DoubleType(), True),
        StructField("atomization_energy_unit", StringType(), True),
        StructField("atomization_energy_per_atom", BooleanType(), True),
        StructField("atomization_energy_reference", DoubleType(), True),
        StructField("atomization_energy_reference_unit", StringType(), True),
        StructField("atomization_energy_property_id", StringType(), True),
    ]
)


dataset_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", LongType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("nconfigurations", IntegerType(), True),
        StructField("nproperty_objects", IntegerType(), True),
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
        StructField("formation_energy_count", IntegerType(), True),
        StructField("free_energy_count", IntegerType(), True),
        StructField("potential_energy_count", IntegerType(), True),
        StructField("atomic_forces_count", IntegerType(), True),
        StructField("band_gap_count", IntegerType(), True),
        StructField("cauchy_stress_count", IntegerType(), True),
        StructField("authors", StringType(), True),  # ArrayType(StringType())
        StructField("description", StringType(), True),
        StructField("extended_id", StringType(), True),
        StructField("license", StringType(), True),
        StructField("publication_link", StringType(), True),  # ArrayType(StringType())
        StructField("data_link", StringType(), True),  # ArrayType(StringType()
        StructField("other_links", StringType(), True),  # ArrayType(StringType()
        StructField("labels", StringType(), True),  # ArrayType(StringType()
        StructField("name", StringType(), True),
    ]
)

dataset_df_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", LongType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("nconfigurations", IntegerType(), True),
        StructField("nproperty_objects", IntegerType(), True),
        StructField("nsites", IntegerType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("elements", ArrayType(StringType()), True),
        StructField("total_elements_ratios", ArrayType(DoubleType()), True),
        StructField("nperiodic_dimensions", ArrayType(IntegerType()), True),
        StructField("dimension_types", ArrayType(ArrayType(IntegerType())), True),
        StructField("atomization_energy_count", IntegerType(), True),
        StructField("adsorption_energy_count", IntegerType(), True),
        StructField("formation_energy_count", IntegerType(), True),
        StructField("free_energy_count", IntegerType(), True),
        StructField("potential_energy_count", IntegerType(), True),
        StructField("atomic_forces_count", IntegerType(), True),
        StructField("band_gap_count", IntegerType(), True),
        StructField("cauchy_stress_count", IntegerType(), True),
        StructField("authors", ArrayType(StringType()), True),
        StructField("description", StringType(), True),
        StructField("extended_id", StringType(), True),
        StructField("license", StringType(), True),
        StructField("publication_link", ArrayType(StringType()), True),
        StructField("data_link", ArrayType(StringType()), True),
        StructField("other_links", ArrayType(StringType()), True),
        StructField("labels", ArrayType(StringType()), True),
        StructField("name", StringType(), True),
    ]
)

configuration_set_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", LongType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("nconfigurations", IntegerType(), True),
        StructField("nsites", IntegerType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("elements", StringType(), True),  # ArrayType(StringType()),
        StructField(
            "total_elements_ratios", StringType(), True
        ),  # ArrayType(DoubleType())
        StructField("description", StringType(), False),
        StructField("name", StringType(), False),
        StructField("dataset_id", StringType(), True),
    ]
)

configuration_set_df_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", LongType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("nconfigurations", IntegerType(), True),
        StructField("nsites", IntegerType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("elements", ArrayType(StringType()), True),
        StructField("dataset_id", ArrayType(StringType()), True),
    ]
)
