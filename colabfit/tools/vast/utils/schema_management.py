import pyarrow as pa
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


def get_spark_field_type(schema: StructType, field_name: str) -> DataType:
    for field in schema:
        if field.name == "$row_id":
            return LongType()
        if field.name == field_name:
            return field.dataType
    raise ValueError(f"Field name {field_name} not found in schema")


def get_stringified_schema(schema: StructType) -> StructType:
    new_fields = []
    for field in schema:
        if field.dataType.typeName() == "array":
            new_fields.append(StructField(field.name, StringType(), field.nullable))
        else:
            new_fields.append(field)
    return StructType(new_fields)


def spark_to_arrow_type(spark_type: DataType) -> pa.DataType:
    """
    Convert PySpark type to PyArrow type.
    Do not include field.nullable, as this conflicts with vastdb-sdk
    """
    if isinstance(spark_type, IntegerType):
        return pa.int32()
    elif isinstance(spark_type, LongType):
        return pa.int64()
    elif isinstance(spark_type, DoubleType):
        return pa.float64()
    elif isinstance(spark_type, StringType):
        return pa.string()
    elif isinstance(spark_type, TimestampType):
        return pa.timestamp("us")
    elif isinstance(spark_type, BooleanType):
        return pa.bool_()
    elif isinstance(spark_type, ArrayType):
        element_type = spark_type.elementType
        if isinstance(element_type, StringType):
            return pa.list_(pa.string())
        elif isinstance(element_type, BooleanType):
            return pa.list_(pa.bool_())
        elif isinstance(element_type, DoubleType):
            return pa.list_(pa.float64())
        elif isinstance(element_type, IntegerType):
            return pa.list_(pa.int32())
        elif isinstance(element_type, ArrayType):
            return pa.list_(spark_to_arrow_type(element_type))
        else:
            raise ValueError(f"Unsupported array element type: {element_type}")
    elif isinstance(spark_type, StructType):
        return pa.schema(
            [
                pa.field(field.name, spark_to_arrow_type(field.dataType))
                for field in spark_type
            ]
        )
    else:
        raise ValueError(f"Unsupported type: {spark_type}")


def spark_schema_to_arrow_schema(spark_schema: StructType) -> pa.Schema:
    """
    Convert PySpark schema to a PyArrow Schema.
    """
    fields = []
    for field in spark_schema:
        if field.name == "$row_id":
            fields.append(pa.field(field.name, pa.uint64()))
        else:
            fields.append(pa.field(field.name, spark_to_arrow_type(field.dataType)))
    arrow_schema = pa.schema(fields)
    for field in arrow_schema:
        field = field.with_nullable(True)
    return arrow_schema


def _empty_dict_from_schema(schema: StructType) -> dict:
    empty_dict = {}
    for field in schema:
        empty_dict[field.name] = None
    return empty_dict
