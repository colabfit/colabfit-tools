from pyspark.sql import DataFrame
from pyspark.sql import functions as sf


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
