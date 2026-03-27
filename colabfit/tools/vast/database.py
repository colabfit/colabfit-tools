import gc
import itertools
import logging
from functools import partial
from itertools import islice
from multiprocessing import Pool
from time import time
from types import GeneratorType

import pyarrow as pa
import pyarrow.compute as pc
from ibis import _
from tqdm import tqdm
from vastdb.session import Session

from colabfit.tools.vast.configuration import AtomicConfiguration
from colabfit.tools.vast.configuration_set import ConfigurationSet
from colabfit.tools.vast.dataset import Dataset
from colabfit.tools.vast.parquet_writer import ParquetWriter
from colabfit.tools.vast.property import Property
from colabfit.tools.vast.schema import (
    co_cs_map_schema,
    config_prop_schema,
    configuration_set_schema,
    dataset_schema,
)
from colabfit.tools.vast.utils import (
    _hash,
    get_last_modified,
)

logger = logging.getLogger(__name__)

VAST_BUCKET_DIR = "colabfit-data"
NSITES_COL_SPLITS = 20
_CONFIGS_COLLECTION = "test_configs"
_CONFIGSETS_COLLECTION = "test_config_sets"
_DATASETS_COLLECTION = "test_datasets"
_CO_CS_MAP_COLLECTION = "test_co_cs_map"
PQ_COMPRESSION_LEVEL = 18


class VastDataLoader:
    def __init__(
        self,
        table_prefix: str = "ndb.colabfit.dev",
        endpoint: str = None,
        access_key: str = None,
        access_secret: str = None,
    ):
        self.table_prefix = table_prefix
        if endpoint and access_key and access_secret:
            self.endpoint = endpoint
            self.access_key = access_key
            self.access_secret = access_secret
            self.session = self.get_vastdb_session(
                endpoint=self.endpoint,
                access_key=self.access_key,
                access_secret=self.access_secret,
            )
        self.config_table = f"{self.table_prefix}.{_CONFIGS_COLLECTION}"
        self.config_set_table = f"{self.table_prefix}.{_CONFIGSETS_COLLECTION}"
        self.dataset_table = f"{self.table_prefix}.{_DATASETS_COLLECTION}"
        self.co_cs_map_table = f"{self.table_prefix}.{_CO_CS_MAP_COLLECTION}"
        self.bucket_dir = VAST_BUCKET_DIR

    def get_vastdb_session(self, endpoint: str, access_key: str, access_secret: str):
        return Session(endpoint=endpoint, access=access_key, secret=access_secret)

    def set_vastdb_session(self, endpoint: str, access_key: str, access_secret: str):
        self.session = self.get_vastdb_session(endpoint, access_key, access_secret)
        self.access_key = access_key
        self.access_secret = access_secret
        self.endpoint = endpoint

    def _get_table_split(self, table_name_str: str):
        """Get bucket, schema and table names for VastDB SDK with no backticks"""
        table_split = table_name_str.split(".")
        bucket_name = table_split[1].replace("`", "")
        schema_name = table_split[2].replace("`", "")
        table_name = table_split[3].replace("`", "")
        return (bucket_name, schema_name, table_name)

    def _table_exists(self, table_name: str) -> bool:
        """Check table existence via VastDB SDK."""
        bucket_name, schema_name, table_n = self._get_table_split(table_name)
        with self.session.transaction() as tx:
            tables = (
                tx.bucket(bucket_name).schema(schema_name).tables(table_name=table_n)
            )
            return len(tables) > 0

    def delete_from_table(self, table_name: str, ids: list[str]):
        if isinstance(ids, str):
            ids = [ids]
        bucket_name, schema_name, table_n = self._get_table_split(table_name)
        with self.session.transaction() as tx:
            table = tx.bucket(bucket_name).schema(schema_name).table(table_n)
            rec_batch = table.select(
                predicate=table["id"].isin(ids), internal_row_id=True
            )
            for batch in rec_batch:
                table.delete(rows=batch)

    def check_unique_ids(self, table_name: str, table: pa.Table):
        if not self._table_exists(table_name):
            logger.info(f"Table {table_name} does not yet exist.")
            return True
        ids = table["id"].to_pylist()
        bucket_name, schema_name, table_n = self._get_table_split(table_name)
        with self.session.transaction() as tx:
            vdb_table = tx.bucket(bucket_name).schema(schema_name).table(table_n)
            for id_batch in tqdm(
                batched(ids, 10000), desc=f"Checking for duplicates in {table_name}"
            ):
                rec_batch_reader = vdb_table.select(
                    predicate=vdb_table["id"].isin(id_batch), columns=["id"]
                )
                for batch in rec_batch_reader:
                    if batch.num_rows > 0:
                        logger.info(f"Duplicate IDs found in table {table_name}")
                        return False
        return True

    def write_table_first(self, table: pa.Table, table_name: str):
        """Write new rows; return existing rows (for multiplicity update)."""
        identifier_dict = {
            self.config_table: "property_hash",
            self.config_set_table: "id",
            self.dataset_table: "id",
            self.co_cs_map_table: "id",
        }
        ider = identifier_dict[table_name]
        arrow_schema = self.get_table_arrow_schema(table_name)
        bucket_name, schema_name, table_n = self._get_table_split(table_name)
        if not self._table_exists(table_name):
            self.create_vastdb_table(table_name, arrow_schema)
        ids = table[ider].to_pylist()
        existing_tables = []
        with self.session.transaction() as tx:
            vdb_table = tx.bucket(bucket_name).schema(schema_name).table(table_n)
            total_rows = 0
            for id_batch in batched(ids, 10000):
                id_batch = list(set(id_batch))
                rec_batch = vdb_table.select(
                    predicate=vdb_table[ider].isin(id_batch),
                    columns=[ider],
                    internal_row_id=False,
                )
                rec_batch = rec_batch.read_all()
                if rec_batch.num_rows > 0:
                    existing_ids = rec_batch[ider].to_pylist()
                    new_ids = [i for i in id_batch if i not in existing_ids]
                    existing_tables.append(
                        table.filter(pc.is_in(table[ider], pa.array(existing_ids)))
                    )
                    if len(new_ids) == 0:
                        logger.info(f"No new IDs to insert into table {table_name}")
                        continue
                    write_rows = table.filter(pc.is_in(table[ider], pa.array(new_ids)))
                else:
                    logger.info(f"No matching IDs found in table {table_name}")
                    write_rows = table
                write_rows = write_rows.select(arrow_schema.names)
                for rec_batch in write_rows.to_batches():
                    vdb_table.insert(rec_batch)
                    total_rows += rec_batch.num_rows
            logger.info(f"Inserted {total_rows} rows into table {table_name}")
        if len(existing_tables) > 1:
            return pa.concat_tables(existing_tables)
        elif len(existing_tables) == 1:
            return existing_tables[0]
        return None

    def write_table_no_check(self, table: pa.Table, table_name: str):
        logger.info(f"Writing table {table_name} without checking for duplicates")
        arrow_schema = self.get_table_arrow_schema(table_name)
        bucket_name, schema_name, table_n = self._get_table_split(table_name)
        if not self._table_exists(table_name):
            self.create_vastdb_table(table_name, arrow_schema)
        table = table.select(arrow_schema.names)
        total_rows = 0
        with self.session.transaction() as tx:
            vdb_table = tx.bucket(bucket_name).schema(schema_name).table(table_n)
            for rec_batch in table.to_batches():
                vdb_table.insert(rec_batch)
                total_rows += rec_batch.num_rows
        logger.info(f"Inserted {total_rows} rows into table {table_name}")

    def create_vastdb_table(self, table_name: str, arrow_schema: pa.Schema):
        logger.info(f"Creating VastDB table {table_name}")
        bucket_name, schema_name, table_n = self._get_table_split(table_name)
        with self.session.transaction() as tx:
            schema = tx.bucket(bucket_name).schema(schema_name)
            schema.create_table(table_n, arrow_schema)
        logger.info(f"Created VastDB table {table_name}")

    def get_table_arrow_schema(self, table_name: str) -> pa.Schema:
        schema_dict = {
            self.config_table: config_prop_schema,
            self.config_set_table: configuration_set_schema,
            self.dataset_table: dataset_schema,
            self.co_cs_map_table: co_cs_map_schema,
        }
        return schema_dict[table_name]

    def write_table(
        self,
        table: pa.Table,
        table_name: str,
        check_unique: bool = True,
    ):
        """Write table to VastDB, optionally checking for duplicate IDs."""
        arrow_schema = self.get_table_arrow_schema(table_name)
        bucket_name, schema_name, table_n = self._get_table_split(table_name)
        table = table.select(arrow_schema.names)
        if not self._table_exists(table_name):
            self.create_vastdb_table(table_name, arrow_schema)
        total_rows = 0
        with self.session.transaction() as tx:
            vdb_table = tx.bucket(bucket_name).schema(schema_name).table(table_n)
            for rec_batch in table.to_batches():
                if check_unique:
                    id_batch = rec_batch["id"].to_pylist()
                    id_rec_batch = vdb_table.select(
                        predicate=vdb_table["id"].isin(id_batch), columns=["id"]
                    )
                    id_rec_batch = id_rec_batch.read_all()
                    if id_rec_batch.num_rows > 0:
                        logger.info(f"Duplicate IDs found in table {table_name}")
                        raise ValueError("Duplicate IDs found in table. Not writing.")
                vdb_table.insert(rec_batch)
                total_rows += rec_batch.num_rows
        logger.info(f"Inserted {total_rows} rows into table {table_name}")

    def increment_multiplicity(
        self, new_table: pa.Table, existing_table: pa.Table
    ) -> pa.Table:
        hash_to_add = dict(
            zip(
                new_table["property_hash"].to_pylist(),
                new_table["multiplicity"].to_pylist(),
            )
        )
        new_mult = [
            (em or 0) + (hash_to_add.get(h, 0) or 0)
            for h, em in zip(
                existing_table["property_hash"].to_pylist(),
                existing_table["multiplicity"].to_pylist(),
            )
        ]
        return existing_table.set_column(
            existing_table.schema.get_field_index("multiplicity"),
            pa.field("multiplicity", pa.int32()),
            pa.array(new_mult, type=pa.int32()),
        )

    def update_existing_co_po_rows(self, df: pa.Table, table_name: str):
        """
        Updates existing rows in CO or PO table with incremented multiplicity.

        Parameters:
        -----------
        df : pa.Table
            Table containing only rows to be updated (identical ids exist).
        table_name : str
            The name of the table to be updated.
        """
        update_cols = ["multiplicity", "last_modified"]
        ids = df["property_hash"].to_pylist()
        bucket_name, schema_name, table_n = self._get_table_split(table_name)
        n_updated = 0
        for id_batch in batched(ids, 10000):
            len_batch = len(id_batch)
            id_batch = list(set(id_batch))
            assert len_batch == len(id_batch)
            with self.session.transaction() as tx:
                vdb_table = tx.bucket(bucket_name).schema(schema_name).table(table_n)
                rec_batch = vdb_table.select(
                    predicate=vdb_table["property_hash"].isin(id_batch),
                    columns=["multiplicity", "property_hash"],
                    internal_row_id=True,
                )
                existing_table = rec_batch.read_all()
            existing_table = self.increment_multiplicity(df, existing_table)
            update_time = get_last_modified()
            update_table = pa.table(
                {
                    "property_hash": existing_table["property_hash"],
                    "multiplicity": existing_table["multiplicity"],
                    "last_modified": pa.array(
                        [update_time] * existing_table.num_rows,
                        type=pa.timestamp("us"),
                    ),
                    "$row_id": existing_table["$row_id"],
                }
            )
            with self.session.transaction() as tx:
                vdb_table = tx.bucket(bucket_name).schema(schema_name).table(table_n)
                vdb_table.update(rows=update_table, columns=update_cols)
            n_updated += len(id_batch)
        assert n_updated == len(ids)
        logger.info(f"Updated {n_updated} rows in {table_name}")

    def zero_multiplicity(self, dataset_id: str):
        """Return multiplicity of POs for a given dataset to zero."""
        if not self._table_exists(self.config_table):
            logger.info(f"Table {self.config_table} does not exist")
            return
        bucket_name, schema_name, table_n = self._get_table_split(self.config_table)
        with self.session.transaction() as tx:
            vdb_table = tx.bucket(bucket_name).schema(schema_name).table(table_n)
            rec_batches = vdb_table.select(
                predicate=(vdb_table["dataset_id"] == dataset_id)
                & (vdb_table["multiplicity"] > 0),
                columns=["property_hash", "multiplicity", "last_modified"],
                internal_row_id=True,
            )
            for rec_batch in rec_batches:
                n = rec_batch.num_rows
                update_time = get_last_modified()
                update_table = pa.table(
                    {
                        "property_hash": rec_batch["property_hash"],
                        "multiplicity": pa.array([0] * n, type=pa.int32()),
                        "last_modified": pa.array(
                            [update_time] * n, type=pa.timestamp("us")
                        ),
                        "$row_id": rec_batch["$row_id"],
                    }
                )
                logger.info(f"Zeroed {n} property objects")
                vdb_table.update(
                    rows=update_table,
                    columns=["multiplicity", "last_modified"],
                )

    def simple_sdk_query(
        self,
        query_table: str,
        predicate,
        internal_row_id: bool = False,
        columns: list[str] = None,
    ) -> pa.Table:
        bucket_name, schema_name, table_n = self._get_table_split(query_table)
        with self.session.transaction() as tx:
            table = tx.bucket(bucket_name).schema(schema_name).table(table_n)
            rec_batch_reader = table.select(
                predicate=predicate,
                internal_row_id=internal_row_id,
                columns=columns,
            )
            result = rec_batch_reader.read_all()
        if result.num_rows == 0:
            logger.info(f"No records found for given query {predicate}")
        return result

    def get_co_cs_mapping(self, cs_id: str):
        """
        Get configuration to configuration set mapping for a given ID.

        Returns pa.Table if found, else None.
        """
        if not self._table_exists(self.co_cs_map_table):
            logger.info(f"Table {self.co_cs_map_table} does not exist")
            return None
        co_cs_map = self.simple_sdk_query(
            self.co_cs_map_table, _.configuration_set_id == cs_id
        )
        if co_cs_map.num_rows == 0:
            logger.info(f"No records found for given configuration set id {cs_id}")
            return None
        return co_cs_map

    def dataset_query(
        self,
        dataset_id: str = None,
        table_name: str = None,
    ) -> pa.Table:
        if dataset_id is None:
            raise ValueError("dataset_id must be provided")
        return self.simple_sdk_query(table_name, _.dataset_id == dataset_id)

    def config_set_query(
        self,
        dataset_id: str = None,
        name_match: str = None,
        label_match: str = None,
    ) -> pa.Table:
        if dataset_id is None:
            raise ValueError("dataset_id must be provided")
        config_df_cols = [
            "property_id",
            "configuration_id",
            "nsites",
            "elements",
            "nperiodic_dimensions",
            "dimension_types",
            "atomic_numbers",
        ]
        fetch_cols = list(config_df_cols)
        if name_match is not None:
            fetch_cols.append("names")
        if label_match is not None:
            fetch_cols.append("labels")
        result = self.simple_sdk_query(
            self.config_table,
            _.dataset_id == dataset_id,
            columns=fetch_cols,
        )
        if name_match is not None:
            names_list = result["names"].to_pylist()
            mask = pa.array([name_match in (names or []) for names in names_list])
            result = result.filter(mask)
        if label_match is not None:
            labels_list = result["labels"].to_pylist()
            mask = pa.array([label_match in (labels or []) for labels in labels_list])
            result = result.filter(mask)
        return result.select(config_df_cols)

    @staticmethod
    def rehash_property_objects(row: dict):
        """
        Rehash property object row after changing values of one or
        more of the columns corresponding to hash_keys defined below.
        """
        hash_keys = [
            "adsorption_energy",
            "atomic_forces",
            "atomization_energy",
            "cauchy_stress",
            "cauchy_stress_volume_normalized",
            "chemical_formula_hill",
            "configuration_id",
            "dataset_id",
            "electronic_band_gap",
            "electronic_band_gap_type",
            "energy",
            "formation_energy",
            "metadata",
            "method",
            "software",
        ]
        row = dict(row)
        row["last_modified"] = get_last_modified()
        row["hash"] = _hash(row, hash_keys, include_keys_in_hash=False)
        if row["cauchy_stress"] is not None:
            row["cauchy_stress"] = str(row["cauchy_stress"])
        id = f'PO_{row["hash"]}'
        if len(id) > 28:
            id = id[:28]
        row["id"] = id
        return {k: v for k, v in row.items() if k != "atomic_forces"}

    @staticmethod
    def config_structure_hash(row: dict, hash_keys: list[str]):
        """
        Rehash configuration object row after changing values of one or
        more of the columns corresponding to hash_keys.
        """
        row = dict(row)
        row["last_modified"] = get_last_modified()
        row["hash"] = _hash(row, hash_keys, include_keys_in_hash=False)
        return row["hash"]


def batched(configs: GeneratorType | list[AtomicConfiguration], n: int):
    "Batch data into tuples of length n. The last batch may be shorter."
    if not isinstance(configs, GeneratorType):
        configs = iter(configs)
    while True:
        batch = list(islice(configs, n))
        if len(batch) == 0:
            break
        yield batch


class DataManager:
    def __init__(
        self,
        nprocs: int = 1,
        configs: list[AtomicConfiguration] = None,
        prop_defs: list[dict] = None,
        prop_map: dict = None,
        dataset_id: str = None,
        standardize_energy: bool = True,
        read_write_batch_size: int = 10000,
    ):
        self.configs = configs
        if not prop_defs:
            logger.warning("No property definitions provided. Defaulting to empty list.")
            prop_defs = []
        if isinstance(prop_defs, dict):
            prop_defs = [prop_defs]
        self.prop_defs = prop_defs
        self.read_write_batch_size = read_write_batch_size
        self.prop_map = prop_map
        self.nprocs = nprocs
        self.dataset_id = dataset_id
        self.standardize_energy = standardize_energy
        logger.info(f"Dataset ID: {self.dataset_id}")

    @staticmethod
    def _gather_co_po_rows(
        prop_defs: list[dict],
        prop_map: dict,
        dataset_id: str,
        configs: list[AtomicConfiguration],
        standardize_energy: bool = True,
    ):
        """Convert COs and POs to row dicts."""
        co_po_rows = []
        for config in configs:
            config.set_dataset_id(dataset_id)
            property = Property.from_definition(
                definitions=prop_defs,
                configuration=config,
                property_map=prop_map,
                standardize_energy=standardize_energy,
            )
            co_po_rows.append(
                (
                    config.row_dict,
                    property.row_dict,
                )
            )
        return co_po_rows

    def get_example_row(self):
        """Returns a pair of (config_row, property_row) for testing ingest scripts."""
        if isinstance(self.configs, list):
            config = [self.configs[0]]
        elif isinstance(self.configs, GeneratorType):
            config = [next(self.configs)]
        else:
            raise ValueError("configs must be a list or generator")
        return (
            config[0],
            self._gather_co_po_rows(
                self.prop_defs,
                self.prop_map,
                self.dataset_id,
                config,
                standardize_energy=self.standardize_energy,
            )[0],
        )

    def gather_co_po_rows_pool(
        self, config_chunks: list[list[AtomicConfiguration]], pool
    ):
        """
        Convert COs and POs to row dicts using multiprocessing Pool.
        Returns a batch of tuples of (configuration_row, property_row).
        """
        part_gather = partial(
            self._gather_co_po_rows,
            self.prop_defs,
            self.prop_map,
            self.dataset_id,
            self.standardize_energy,
        )
        return itertools.chain.from_iterable(pool.map(part_gather, list(config_chunks)))

    def gather_co_po_in_batches(self):
        """
        Yields batches of CO-PO rows using multiprocessing pool,
        preventing configuration iterator from being consumed all at once.
        """
        chunk_size = 1000
        config_chunks = batched(self.configs, chunk_size)

        with Pool(self.nprocs) as pool:
            while True:
                config_batches = list(islice(config_chunks, self.nprocs))
                if not config_batches:
                    break
                else:
                    yield list(self.gather_co_po_rows_pool(config_batches, pool))

    def gather_co_po_in_batches_no_pool(self):
        """
        Yields batches of CO-PO rows without multiprocessing,
        preventing configuration iterator from being consumed all at once.
        """
        chunk_size = self.read_write_batch_size
        config_chunks = batched(self.configs, chunk_size)
        for chunk in config_chunks:
            yield list(
                self._gather_co_po_rows(
                    self.prop_defs,
                    self.prop_map,
                    self.dataset_id,
                    chunk,
                    standardize_energy=self.standardize_energy,
                )
            )

    def deduplicate_co_po_rows(self, rows: list[dict]) -> list[dict]:
        """Aggregate multiplicity for rows with duplicate property_hash values."""
        seen = {}
        for row in rows:
            h = row["property_hash"]
            if h not in seen:
                seen[h] = {**row, "multiplicity": 1}
            else:
                seen[h]["multiplicity"] += 1
        return list(seen.values())

    def check_existing_tables(
        self, loader: VastDataLoader, batching_ingest: bool = False
    ):
        """Check tables for conflicts before loading data."""
        if loader._table_exists(loader.config_table):
            logger.info(f"table {loader.config_table} exists")
            if batching_ingest is False:
                cos_with_mult = loader.simple_sdk_query(
                    loader.config_table,
                    (_.dataset_id == self.dataset_id) & (_.multiplicity > 0),
                    columns=["dataset_id", "multiplicity"],
                )
                if cos_with_mult.num_rows > 0:
                    raise ValueError(
                        f"COs for dataset with ID {self.dataset_id} already exist in "
                        "database with multiplicity > 0.\nTo continue, set "
                        "multiplicities to 0 with "
                        f'loader.zero_multiplicity("{self.dataset_id}")'
                    )
        if loader._table_exists(loader.dataset_table):
            dataset_row = loader.simple_sdk_query(
                loader.dataset_table,
                _.id == self.dataset_id,
                columns=["id"],
            )
            if dataset_row.num_rows > 0:
                raise ValueError(f"Dataset with ID {self.dataset_id} already exists.")

    def combine_co_po_rows(self, co_po_rows: list[dict]):
        """Combine configuration and property rows into a single row."""
        combined_rows = []
        for co_row, po_row in co_po_rows:
            po_row["property_id"] = po_row.pop("id")
            co_row["configuration_id"] = co_row.pop("id")
            po_row["property_hash"] = po_row.pop("hash")
            co_row["configuration_hash"] = co_row.pop("hash")
            co_po_row = {
                k: v
                for k, v in {**co_row, **po_row}.items()
                if k in config_prop_schema.names
            }
            for key, value in co_po_row.items():
                if value is not None and hasattr(value, "__len__"):
                    if len(str(value)) > 50000:
                        logger.warning(
                            f"Large value in {key}: {len(str(value))} characters"
                        )
            combined_rows.append(co_po_row)
        return combined_rows

    def load_co_po_to_vastdb(
        self,
        loader: VastDataLoader,
        batching_ingest: bool = False,
        check_existing: bool = False,
        parquet_writer: ParquetWriter = None,
    ):
        self.check_existing_tables(loader, batching_ingest)
        co_po_rows = self.gather_co_po_in_batches_no_pool()
        for co_po_batch in tqdm(
            co_po_rows,
            desc="Loading data to database: ",
            unit="batch",
        ):
            co_po_combined_rows = self.combine_co_po_rows(co_po_batch)
            if parquet_writer is not None:
                parquet_writer.add_co_rows(co_po_combined_rows)
            del co_po_batch
            co_count = len(co_po_combined_rows)
            if co_count == 0:
                continue
            for i, row in enumerate(co_po_combined_rows[:3]):
                if not isinstance(row, dict):
                    raise ValueError(f"Row {i} is not a dict: {type(row)}")
                for key, value in row.items():
                    if value is None:
                        continue
                    if hasattr(value, "__len__") and len(str(value)) > 100000:
                        logger.warning(
                            f"Very large value in {key}: {len(str(value))} chars"
                        )

            co_table = pa.Table.from_pylist(
                co_po_combined_rows, schema=config_prop_schema
            )
            del co_po_combined_rows
            co_count_distinct = len(pc.unique(co_table["property_hash"].drop_null()))
            if co_count_distinct < co_count:
                logger.info(
                    f"{co_count - co_count_distinct} duplicates found in CO-PO table"
                )
                dedup_rows = self.deduplicate_co_po_rows(co_table.to_pylist())
                co_table = pa.Table.from_pylist(dedup_rows, schema=config_prop_schema)
                del dedup_rows
            if check_existing:
                co_existing = loader.write_table_first(co_table, loader.config_table)
                if co_existing is not None:
                    loader.update_existing_co_po_rows(
                        df=co_existing,
                        table_name=loader.config_table,
                    )
                logger.info(f"Wrote {co_count} rows to {loader.config_table}")
                del co_existing
            else:
                loader.write_table_no_check(co_table, loader.config_table)
                logger.info(f"Wrote {co_count} rows to {loader.config_table}")
                del co_table
        if parquet_writer:
            parquet_writer.write_final("co")
        logger.info("Garbage collecting")
        gc.collect()

    def create_configuration_sets(
        self,
        loader: VastDataLoader,
        name_label_match: list[tuple],
        parquet_writer: ParquetWriter = None,
    ):
        """
        Args for name_label_match in order:
        1. String pattern for matching CONFIGURATION NAMES
        2. String pattern for matching CONFIGURATION LABELS
        3. Name for configuration set
        4. Description for configuration set
        5. Boolean for whether configuration set is ordered
        """
        dataset_id = self.dataset_id
        config_set_rows = []
        co_cs_write_df = None
        for i, (names_match, label_match, cs_name, cs_desc, ordered) in tqdm(
            enumerate(name_label_match), desc="Creating Configuration Sets"
        ):
            logger.info(
                f"names match: {names_match}, label: {label_match}, "
                f"cs_name: {cs_name}, cs_desc: {cs_desc}, ordered: {ordered}"
            )
            config_df = loader.config_set_query(
                dataset_id=dataset_id,
                name_match=names_match,
                label_match=label_match,
            )
            co_ids = pc.unique(config_df["configuration_id"].drop_null())
            t = time()
            prelim_cs_id = f"CS_{cs_name}_{self.dataset_id}"
            co_cs_exists = loader.get_co_cs_mapping(prelim_cs_id)
            if co_cs_exists:
                logger.error(
                    f"Configuration Set {cs_name} already exists.\n"
                    f"Remove rows matching 'configuration_set_id == {prelim_cs_id}' "
                    f"from table {loader.co_cs_map_table} to recreate.\n"
                )
                continue
            config_set = ConfigurationSet(
                name=cs_name,
                description=cs_desc,
                config_df=config_df,
                dataset_id=self.dataset_id,
                ordered=ordered,
            )
            co_ids_table = pa.table(
                {
                    "configuration_id": co_ids,
                    "configuration_set_id": pa.array([config_set.id] * len(co_ids)),
                }
            )
            if parquet_writer:
                parquet_writer.add_co_cs_map_rows(co_ids_table.to_pylist())
                parquet_writer.add_cs_rows([config_set.row_dict])
                continue
            if co_cs_write_df is None:
                co_cs_write_df = co_ids_table
            elif co_cs_write_df.num_rows > 10000:
                logger.info("Sending CO-CS map batch to write table")
                loader.write_table(
                    co_cs_write_df, loader.co_cs_map_table, check_unique=False
                )
                co_cs_write_df = co_ids_table
            else:
                co_cs_write_df = pa.concat_tables([co_cs_write_df, co_ids_table])
            config_set_rows.append(config_set.row_dict)
            logger.info(f"Time to create CS: {time() - t}")
        if co_cs_write_df is not None and co_cs_write_df.num_rows > 0:
            loader.write_table(
                co_cs_write_df, loader.co_cs_map_table, check_unique=False
            )
        config_set_table = pa.Table.from_pylist(
            config_set_rows, schema=configuration_set_schema
        )
        loader.write_table(config_set_table, loader.config_set_table)
        if parquet_writer:
            parquet_writer.write_final("cs")
            parquet_writer.write_final("co_cs_map")
        return config_set_rows

    def create_dataset(
        self,
        loader: VastDataLoader,
        name: str,
        authors: list[str],
        description: str,
        publication_link: str,
        data_link: str,
        other_links: list[str] = None,
        publication_year: str = None,
        doi: str = None,
        labels: list[str] = None,
        equilibrium: bool = False,
        date_requested: str = None,
        data_license: str = "CC-BY-4.0",
        parquet_writer: ParquetWriter = None,
    ):
        if loader._table_exists(loader.config_set_table):
            cs_ids_table = loader.dataset_query(
                dataset_id=self.dataset_id, table_name=loader.config_set_table
            )
            cs_ids = cs_ids_table["id"].to_pylist()
            if len(cs_ids) == 0:
                cs_ids = None
        else:
            cs_ids = None
        logger.info(f"Configuration Set IDs: {cs_ids}")
        config_df = loader.dataset_query(
            dataset_id=self.dataset_id, table_name=loader.config_table
        )
        ds = Dataset(
            name=name,
            authors=authors,
            config_df=config_df,
            publication_link=publication_link,
            data_link=data_link,
            description=description,
            other_links=other_links,
            dataset_id=self.dataset_id,
            labels=labels,
            doi=doi,
            data_license=data_license,
            configuration_set_ids=cs_ids,
            publication_year=publication_year,
            date_requested=date_requested,
            equilibrium=equilibrium,
        )
        if parquet_writer:
            parquet_writer.write_ds_parquet(ds.row_dict)
            return
        ds_table = pa.Table.from_pylist([ds.row_dict], schema=dataset_schema)
        loader.write_table(ds_table, loader.dataset_table)
