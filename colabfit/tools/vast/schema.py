import pyarrow as pa

NSITES_COL_SPLITS = 20

config_prop_schema = pa.schema(
    [
        pa.field("property_id", pa.string()),
        pa.field("property_hash", pa.string()),
        pa.field("last_modified", pa.timestamp("us")),
        pa.field("dataset_id", pa.string()),
        pa.field("multiplicity", pa.int32()),
        pa.field("software", pa.string()),
        pa.field("method", pa.string()),
        pa.field("energy", pa.float64()),
        pa.field("atomic_forces", pa.list_(pa.list_(pa.float64()))),
        pa.field("cauchy_stress", pa.list_(pa.list_(pa.float64()))),
        pa.field("cauchy_stress_volume_normalized", pa.bool_()),
        pa.field("electronic_band_gap", pa.float64()),
        pa.field("electronic_band_gap_type", pa.string()),
        pa.field("formation_energy", pa.float64()),
        pa.field("adsorption_energy", pa.float64()),
        pa.field("atomization_energy", pa.float64()),
        pa.field("max_force_norm", pa.float64()),
        pa.field("mean_force_norm", pa.float64()),
        pa.field("energy_above_hull", pa.float64()),
        pa.field("configuration_id", pa.string()),
        pa.field("configuration_hash", pa.string()),
        pa.field("structure_hash", pa.string()),
        pa.field("cell", pa.list_(pa.list_(pa.float64()))),
        pa.field("positions", pa.list_(pa.list_(pa.float64()))),
        pa.field("pbc", pa.list_(pa.bool_())),
        pa.field("chemical_formula_hill", pa.string()),
        pa.field("chemical_formula_reduced", pa.string()),
        pa.field("chemical_formula_anonymous", pa.string()),
        pa.field("elements", pa.list_(pa.string())),
        pa.field("elements_ratios", pa.list_(pa.float64())),
        pa.field("atomic_numbers", pa.list_(pa.int32())),
        pa.field("nsites", pa.int32()),
        pa.field("nelements", pa.int32()),
        pa.field("nperiodic_dimensions", pa.int32()),
        pa.field("dimension_types", pa.list_(pa.int32())),
        pa.field("names", pa.list_(pa.string())),
        pa.field("labels", pa.list_(pa.string())),
        pa.field("metadata", pa.string()),
        pa.field("has_forces", pa.bool_()),
        pa.field("has_stress", pa.bool_()),
        pa.field("element_filter", pa.string()),
    ]
)

config_schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("hash", pa.string()),
        pa.field("last_modified", pa.timestamp("us")),
        pa.field("dataset_ids", pa.list_(pa.string())),
        pa.field("chemical_formula_hill", pa.string()),
        pa.field("chemical_formula_reduced", pa.string()),
        pa.field("chemical_formula_anonymous", pa.string()),
        pa.field("elements", pa.list_(pa.string())),
        pa.field("elements_ratios", pa.list_(pa.float64())),
        pa.field("atomic_numbers", pa.list_(pa.int32())),
        pa.field("nsites", pa.int32()),
        pa.field("nelements", pa.int32()),
        pa.field("nperiodic_dimensions", pa.int32()),
        pa.field("cell", pa.list_(pa.list_(pa.float64()))),
        pa.field("dimension_types", pa.list_(pa.int32())),
        pa.field("pbc", pa.list_(pa.bool_())),
        pa.field("names", pa.list_(pa.string())),
        pa.field("labels", pa.list_(pa.string())),
        pa.field("structure_hash", pa.string()),
        pa.field("positions", pa.list_(pa.list_(pa.float64()))),
    ]
)
config_row_id_schema = config_schema.append(pa.field("$row_id", pa.uint64()))

property_object_schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("hash", pa.string()),
        pa.field("last_modified", pa.timestamp("us")),
        pa.field("configuration_id", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("multiplicity", pa.int32()),
        pa.field("software", pa.string()),
        pa.field("method", pa.string()),
        pa.field("chemical_formula_hill", pa.string()),
        pa.field("energy", pa.float64()),
        pa.field("atomic_forces", pa.list_(pa.list_(pa.float64()))),
        pa.field("cauchy_stress", pa.list_(pa.list_(pa.float64()))),
        pa.field("cauchy_stress_volume_normalized", pa.bool_()),
        pa.field("electronic_band_gap", pa.float64()),
        pa.field("electronic_band_gap_type", pa.string()),
        pa.field("energy_above_hull", pa.float64()),
        pa.field("formation_energy", pa.float64()),
        pa.field("adsorption_energy", pa.float64()),
        pa.field("atomization_energy", pa.float64()),
        pa.field("max_force_norm", pa.float64()),
        pa.field("mean_force_norm", pa.float64()),
        pa.field("metadata", pa.string()),
    ]
)
property_object_row_id_schema = property_object_schema.append(
    pa.field("$row_id", pa.uint64())
)

dataset_schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("hash", pa.string()),
        pa.field("name", pa.string()),
        pa.field("last_modified", pa.timestamp("us")),
        pa.field("nconfigurations", pa.int32()),
        pa.field("nproperty_objects", pa.int64()),
        pa.field("nsites", pa.int64()),
        pa.field("nelements", pa.int32()),
        pa.field("elements", pa.list_(pa.string())),
        pa.field("total_elements_ratios", pa.list_(pa.float64())),
        pa.field("nperiodic_dimensions", pa.list_(pa.int32())),
        pa.field("dimension_types", pa.list_(pa.list_(pa.int32()))),
        pa.field("energy_count", pa.int64()),
        pa.field("energy_mean", pa.float64()),
        pa.field("energy_variance", pa.float64()),
        pa.field("atomization_energy_count", pa.int64()),
        pa.field("adsorption_energy_count", pa.int64()),
        pa.field("energy_above_hull_count", pa.int64()),
        pa.field("formation_energy_count", pa.int64()),
        pa.field("atomic_forces_count", pa.int64()),
        pa.field("electronic_band_gap_count", pa.int64()),
        pa.field("cauchy_stress_count", pa.int64()),
        pa.field("authors", pa.list_(pa.string())),
        pa.field("description", pa.string()),
        pa.field("extended_id", pa.string()),
        pa.field("license", pa.string()),
        pa.field("links", pa.string()),
        pa.field("publication_year", pa.string()),
        pa.field("doi", pa.string()),
        pa.field("equilibrium", pa.bool_()),
        pa.field("methods", pa.list_(pa.string())),
        pa.field("software", pa.list_(pa.string())),
        pa.field("date_added_to_colabfit", pa.timestamp("us")),
        pa.field("date_requested", pa.timestamp("us")),
    ]
)

configuration_set_schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("hash", pa.string()),
        pa.field("last_modified", pa.timestamp("us")),
        pa.field("nconfigurations", pa.int32()),
        pa.field("nperiodic_dimensions", pa.list_(pa.int32())),
        pa.field("dimension_types", pa.list_(pa.list_(pa.int32()))),
        pa.field("nsites", pa.int64()),
        pa.field("nelements", pa.int32()),
        pa.field("elements", pa.list_(pa.string())),
        pa.field("total_elements_ratios", pa.list_(pa.float64())),
        pa.field("description", pa.string()),
        pa.field("name", pa.string()),
        pa.field("dataset_id", pa.string()),
        pa.field("ordered", pa.bool_()),
        pa.field("extended_id", pa.string()),
    ]
)

co_cs_map_schema = pa.schema(
    [
        pa.field("configuration_id", pa.string()),
        pa.field("configuration_set_id", pa.string()),
    ]
)
