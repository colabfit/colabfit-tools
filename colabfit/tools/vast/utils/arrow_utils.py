import pyarrow as pa
import pyarrow.compute as pc


def get_max_string_length(table: pa.Table, column_name: str) -> int:
    lengths = pc.utf8_length(table[column_name])
    max_len = pc.max(lengths).as_py()
    if max_len is None:
        return 0
    return max_len
