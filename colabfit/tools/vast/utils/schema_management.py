import pyarrow as pa


def _empty_dict_from_schema(schema: pa.Schema) -> dict:
    return {field.name: None for field in schema}
