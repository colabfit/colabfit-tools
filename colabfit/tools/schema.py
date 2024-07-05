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

from colabfit.tools.utilities import get_stringified_schema

NSITES_COL_SPLITS = 20

config_df_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", StringType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("dataset_ids", ArrayType(StringType()), True),
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
        StructField("names", ArrayType(StringType()), True),
        StructField("labels", ArrayType(StringType()), True),
        StructField("configuration_set_ids", ArrayType(StringType()), True),
        StructField("metadata_id", StringType(), True),
        StructField("metadata_path", StringType(), True),
        StructField("metadata_size", IntegerType(), True),
    ]
    + [
        StructField(f"positions_{i:02d}", ArrayType(ArrayType(DoubleType())), True)
        for i in range(NSITES_COL_SPLITS)
    ]
)
config_schema = get_stringified_schema(config_df_schema)


property_object_df_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", StringType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("configuration_id", StringType(), True),
        StructField("dataset_id", StringType(), True),
        StructField("multiplicity", IntegerType(), True),
        StructField("metadata_id", StringType(), True),
        StructField("metadata_path", StringType(), True),
        StructField("metadata_size", IntegerType(), True),
        StructField("software", StringType(), True),
        StructField("method", StringType(), True),
        StructField("chemical_formula_hill", StringType(), True),
        StructField("energy_conjugate_with_atomic_forces", DoubleType(), True),
        StructField("energy_conjugate_with_atomic_forces_unit", StringType(), True),
        StructField(
            "energy_conjugate_with_atomic_forces_property_id", StringType(), True
        ),
    ]
    + [
        StructField(f"atomic_forces_{i:02d}", ArrayType(ArrayType(DoubleType())), True)
        for i in range(NSITES_COL_SPLITS)
    ]
    + [
        StructField("atomic_forces_unit", StringType(), True),
        StructField("atomic_forces_property_id", StringType(), True),
        StructField("cauchy_stress", ArrayType(ArrayType(DoubleType())), True),
        StructField("cauchy_stress_unit", StringType(), True),
        StructField("cauchy_stress_volume_normalized", BooleanType(), True),
        StructField("cauchy_stress_property_id", StringType(), True),
        StructField("electronic_band_gap", DoubleType(), True),
        StructField("electronic_band_gap_unit", StringType(), True),
        StructField("electronic_band_gap_direct", StringType(), True),
        StructField("electronic_band_gap_type", StringType(), True),
        StructField("electronic_band_gap_property_id", StringType(), True),
        StructField("formation_energy", DoubleType(), True),
        StructField("formation_energy_unit", StringType(), True),
        StructField("formation_energy_property_id", StringType(), True),
        StructField("adsorption_energy", DoubleType(), True),
        StructField("adsorption_energy_unit", StringType(), True),
        StructField("adsorption_energy_property_id", StringType(), True),
        StructField("atomization_energy", DoubleType(), True),
        StructField("atomization_energy_unit", StringType(), True),
        StructField("atomization_energy_property_id", StringType(), True),
    ]
)

property_object_schema = get_stringified_schema(property_object_df_schema)


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
        StructField("energy_conjugate_with_atomic_forces_count", IntegerType(), True),
        StructField("energy_conjugate_with_atomic_forces_mean", DoubleType(), True),
        StructField("energy_conjugate_with_atomic_forces_variance", DoubleType(), True),
        StructField("atomization_energy_count", IntegerType(), True),
        StructField("adsorption_energy_count", IntegerType(), True),
        StructField("formation_energy_count", IntegerType(), True),
        StructField("atomic_forces_count", IntegerType(), True),
        StructField("electronic_band_gap_count", IntegerType(), True),
        StructField("cauchy_stress_count", IntegerType(), True),
        StructField("authors", ArrayType(StringType()), True),
        StructField("description", StringType(), True),
        StructField("extended_id", StringType(), True),
        StructField("license", StringType(), True),
        StructField("publication_link", StringType(), True),
        StructField("data_link", StringType(), True),
        StructField("other_links", ArrayType(StringType()), True),
        StructField("labels", ArrayType(StringType()), True),
        StructField("name", StringType(), True),
    ]
)

dataset_schema = get_stringified_schema(dataset_df_schema)


configuration_set_df_schema = StructType(
    [
        StructField("id", StringType(), False),
        StructField("hash", LongType(), False),
        StructField("last_modified", TimestampType(), False),
        StructField("nconfigurations", IntegerType(), True),
        StructField("nsites", IntegerType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("elements", ArrayType(StringType()), True),
        StructField("total_elements_ratios", ArrayType(DoubleType()), True),
        StructField("description", StringType(), False),
        StructField("name", StringType(), False),
        StructField("dataset_id", StringType(), True),
    ]
)

configuration_set_schema = get_stringified_schema(configuration_set_df_schema)
