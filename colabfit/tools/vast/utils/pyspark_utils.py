from ast import literal_eval

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql.types import ArrayType, IntegerType, StringType, DoubleType, BooleanType

############################################################
# Functions for converting column values to/from string
############################################################


@sf.udf(returnType=StringType())
def stringify_df_val_udf(val):
    if val is not None:
        return str(val)


@sf.udf(returnType=ArrayType(StringType()))
def str_to_arrayof_str(val):
    try:
        if isinstance(val, str) and len(val) > 0 and val[0] == "[":
            return literal_eval(val)
    except ValueError:
        raise ValueError(f"Error converting {val} to list")


@sf.udf(returnType=ArrayType(IntegerType()))
def str_to_arrayof_int(val):
    if isinstance(val, str) and len(val) > 0 and val[0] == "[":
        return literal_eval(val)
    raise ValueError(f"Error converting {val} to list")


@sf.udf(returnType=ArrayType(DoubleType()))
def str_to_arrayof_double(val):
    if isinstance(val, str) and len(val) > 0 and val[0] == "[":
        return literal_eval(val)
    raise ValueError(f"Error converting {val} to list")


@sf.udf(returnType=ArrayType(BooleanType()))
def str_to_arrayof_bool(val):
    try:
        if isinstance(val, str) and len(val) > 0 and val[0] == "[":
            return literal_eval(val)
    except ValueError:
        raise ValueError(f"Error converting {val} to list")


@sf.udf(returnType=ArrayType(ArrayType(DoubleType())))
def str_to_arrayofarrayof_double(val):
    try:
        if isinstance(val, str) and len(val) > 0 and val[0] == "[[":
            return literal_eval(val)
    except ValueError:
        raise ValueError(f"Error converting {val} to list")


############################################################
# Assorted
############################################################


def get_max_string_length(df: DataFrame, column_name: str) -> int:

    max_len = (
        df.select(sf.length(column_name).alias("string_length"))
        .agg(sf.max("string_length"))
        .collect()[0][0]
    )
    if max_len is None:
        return 0
    return max_len
