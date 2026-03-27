import logging
from pathlib import Path

import pyarrow as pa
from pyarrow.parquet import ParquetWriter as _ParquetWriter

from colabfit.tools.vast.schema import (
    co_cs_map_schema,
    config_prop_schema,
    configuration_set_schema,
    dataset_schema,
)

logger = logging.getLogger(__name__)

PQ_COMPRESSION_LEVEL = 18


class ParquetWriter:
    def __init__(self, dataset_id: str, parquet_size: int = 1_000_000):
        self.dataset_id = dataset_id
        self.dir_path = Path(f"./{dataset_id}")
        self.ds_fp = self.dir_path / "ds.parquet"
        self.co_dir = self.dir_path / "co"
        self.cs_dir = self.dir_path / "cs"
        self.co_cs_map_dir = self.dir_path / "co_cs_map"
        self.co_rows = []
        self.co_cs_map_rows = []
        self.cs_rows = []
        self.parquet_size = parquet_size
        self.co_row_count = 0
        self.cs_row_count = 0
        self.co_cs_map_row_count = 0
        self.co_file_count = 0
        self.cs_file_count = 0
        self.co_cs_map_file_count = 0

    def add_co_rows(self, co_rows: list[dict]):
        self.co_rows.extend(co_rows)
        if len(self.co_rows) >= self.parquet_size:
            self.write_parquet("co", config_prop_schema, self.co_rows)
            self.co_rows = []

    def add_cs_rows(self, cs_rows: list[dict]):
        """Add configuration set rows and write to parquet if threshold reached"""
        if len(cs_rows) >= self.parquet_size:
            self.write_parquet("cs", configuration_set_schema, cs_rows)
            self.cs_rows = []
        else:
            self.cs_rows.extend(cs_rows)

    def add_co_cs_map_rows(self, co_cs_map_rows: list[dict]):
        """Add configuration-to-configuration-set mapping rows"""
        if len(co_cs_map_rows) >= self.parquet_size:
            self.write_parquet("co_cs_map", co_cs_map_schema, co_cs_map_rows)
            self.co_cs_map_rows = []
        else:
            self.co_cs_map_rows.extend(co_cs_map_rows)

    def write_parquet(self, row_type: str, schema: pa.Schema, rows: list[dict]):
        """
        General method to write parquet files for co, cs, or co_cs_map data.

        Args:
            row_type (str): Type of data - 'co', 'cs', or 'co_cs_map'
            schema (pa.Schema): Arrow schema for the data
            rows (list[dict]): Rows to write.
        """
        if not rows:
            logger.warning(f"No rows to write for {row_type}. Skipping write.")
            return

        dir_mapping = {
            "co": (self.co_dir, "co_file_count"),
            "cs": (self.cs_dir, "cs_file_count"),
            "co_cs_map": (self.co_cs_map_dir, "co_cs_map_file_count"),
        }

        if row_type not in dir_mapping:
            raise ValueError(
                f"Invalid row_type: {row_type}. Must be one of: co, cs, co_cs_map"
            )

        target_dir, file_count_attr = dir_mapping[row_type]

        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True, exist_ok=True)
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pylist(rows, schema=schema)
        file_count = getattr(self, file_count_attr)
        fp = target_dir / f"{row_type}_{file_count}.parquet"

        with _ParquetWriter(
            fp,
            schema,
            compression="zstd",
            compression_level=PQ_COMPRESSION_LEVEL,
        ) as writer:
            writer.write_table(table)
            setattr(self, file_count_attr, file_count + 1)
            logger.info(f"Wrote {len(rows)} {row_type.upper()} rows to {fp}")

    def write_ds_parquet(self, ds_row: dict):
        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True, exist_ok=True)
        ds_table = pa.Table.from_pylist([ds_row], schema=dataset_schema)
        with _ParquetWriter(
            self.ds_fp,
            dataset_schema,
            compression="zstd",
            compression_level=PQ_COMPRESSION_LEVEL,
        ) as writer:
            writer.write_table(ds_table)
            logger.info(f"Wrote dataset row to {self.ds_fp}")

    def write_final(self, row_type: str):
        """Write any remaining rows to parquet files."""
        schema_map = {
            "co": config_prop_schema,
            "cs": configuration_set_schema,
            "co_cs_map": co_cs_map_schema,
        }
        rows_map = {
            "co": self.co_rows,
            "cs": self.cs_rows,
            "co_cs_map": self.co_cs_map_rows,
        }
        rows = rows_map.get(row_type, [])
        if rows:
            self.write_parquet(row_type, schema_map[row_type], rows)
            rows_map[row_type].clear()
