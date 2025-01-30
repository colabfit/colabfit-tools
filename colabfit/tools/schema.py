"""
Each schema for VastDB has a corresponding schema for a dataframe with
the same fields, but with non-stringified lists.
config_schema (with stringified lists), for example, has a corresponding
config_arr_schema (with non-stringified lists).
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

config_arr_schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("hash", StringType(), True),
        StructField("last_modified", TimestampType(), True),
        StructField("dataset_ids", ArrayType(StringType()), True),
        StructField("configuration_set_ids", ArrayType(StringType()), True),
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
        StructField("metadata_id", StringType(), True),
        StructField("metadata_path", StringType(), True),
        StructField("metadata_size", IntegerType(), True),
        StructField("structure_hash", StringType(), True),
    ]
    + [
        StructField(f"positions_{i:02d}", ArrayType(ArrayType(DoubleType())), True)
        for i in range(NSITES_COL_SPLITS)
    ]
)
config_schema = get_stringified_schema(config_arr_schema)
config_md_schema = config_arr_schema.add(StructField("metadata", StringType(), True))


property_object_arr_schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("hash", StringType(), True),
        StructField("last_modified", TimestampType(), True),
        StructField("configuration_id", StringType(), True),
        StructField("dataset_id", StringType(), True),
        StructField("multiplicity", IntegerType(), True),
        StructField("metadata_id", StringType(), True),
        StructField("metadata_path", StringType(), True),
        StructField("metadata_size", IntegerType(), True),
        StructField("software", StringType(), True),
        StructField("method", StringType(), True),
        StructField("chemical_formula_hill", StringType(), True),
        StructField("energy", DoubleType(), True),
    ]
    + [
        StructField(f"atomic_forces_{i:02d}", ArrayType(ArrayType(DoubleType())), True)
        for i in range(NSITES_COL_SPLITS)
    ]
    + [
        StructField("cauchy_stress", ArrayType(ArrayType(DoubleType())), True),
        StructField("cauchy_stress_volume_normalized", BooleanType(), True),
        StructField("electronic_band_gap", DoubleType(), True),
        StructField("electronic_band_gap_type", StringType(), True),
        StructField("formation_energy", DoubleType(), True),
        StructField("adsorption_energy", DoubleType(), True),
        StructField("atomization_energy", DoubleType(), True),
        StructField("max_force_norm", DoubleType(), True),
        StructField("mean_force_norm", DoubleType(), True),
    ]
)

property_object_schema = get_stringified_schema(property_object_arr_schema)
property_object_md_schema = property_object_arr_schema.add(
    StructField("metadata", StringType(), True)
)


dataset_arr_schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("hash", StringType(), True),
        StructField("name", StringType(), True),
        StructField("last_modified", TimestampType(), True),
        StructField("nconfigurations", IntegerType(), True),
        StructField("nproperty_objects", LongType(), True),
        StructField("nsites", LongType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("elements", ArrayType(StringType()), True),
        StructField("total_elements_ratios", ArrayType(DoubleType()), True),
        StructField("nperiodic_dimensions", ArrayType(IntegerType()), True),
        StructField("dimension_types", ArrayType(ArrayType(IntegerType())), True),
        StructField("energy_count", LongType(), True),
        StructField("energy_mean", DoubleType(), True),
        StructField("energy_variance", DoubleType(), True),
        StructField("atomization_energy_count", LongType(), True),
        StructField("adsorption_energy_count", LongType(), True),
        StructField("formation_energy_count", LongType(), True),
        StructField("atomic_forces_count", LongType(), True),
        StructField("electronic_band_gap_count", LongType(), True),
        StructField("cauchy_stress_count", LongType(), True),
        StructField("authors", ArrayType(StringType()), True),
        StructField("description", StringType(), True),
        StructField("extended_id", StringType(), True),
        StructField("license", StringType(), True),
        StructField("links", StringType(), True),
        StructField("publication_year", StringType(), True),
        StructField("doi", StringType(), True),
        # StructField("labels", ArrayType(StringType()), True),
    ]
)

dataset_schema = get_stringified_schema(dataset_arr_schema)


configuration_set_arr_schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("hash", StringType(), True),
        StructField("last_modified", TimestampType(), True),
        StructField("nconfigurations", IntegerType(), True),
        StructField("nperiodic_dimensions", ArrayType(IntegerType()), True),
        StructField("dimension_types", ArrayType(ArrayType(IntegerType())), True),
        StructField("nsites", LongType(), True),
        StructField("nelements", IntegerType(), True),
        StructField("elements", ArrayType(StringType()), True),
        StructField("total_elements_ratios", ArrayType(DoubleType()), True),
        StructField("description", StringType(), True),
        StructField("name", StringType(), True),
        StructField("dataset_id", StringType(), True),
        StructField("ordered", BooleanType(), True),
        StructField("extended_id", StringType(), True),
    ]
)

configuration_set_schema = get_stringified_schema(configuration_set_arr_schema)

co_cs_map_schema = StructType(
    [
        StructField("configuration_id", StringType(), True),
        StructField("configuration_set_id", StringType(), True),
    ]
)
