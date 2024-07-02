import datetime
import itertools
import multiprocessing
import os
import string
import warnings
from functools import partial
from itertools import islice
from multiprocessing import Pool
from time import time
from types import GeneratorType

import dateutil.parser
import findspark
import psycopg
import pyarrow as pa
import pyspark.sql.functions as sf
from django.utils.crypto import get_random_string
from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from tqdm import tqdm
from vastdb.session import Session

from colabfit import (
    ID_FORMAT_STRING,
)  # ATOMS_NAME_FIELD,; EXTENDED_ID_STRING_NAME,; MAX_STRING_LENGTH,; SHORT_ID_STRING_NAME,; _CONFIGS_COLLECTION,; _CONFIGSETS_COLLECTION,; _DATASETS_COLLECTION,; _PROPOBJECT_COLLECTION,
from colabfit.tools.configuration import AtomicConfiguration
from colabfit.tools.configuration_set import ConfigurationSet
from colabfit.tools.dataset import Dataset
from colabfit.tools.property import Property
from colabfit.tools.schema import (
    config_df_schema,
    config_schema,
    configuration_set_df_schema,
    configuration_set_schema,
    dataset_df_schema,
    dataset_schema,
    property_object_df_schema,
    property_object_schema,
)
from colabfit.tools.utilities import (
    get_spark_field_type,
    spark_schema_to_arrow_schema,
    stringify_df_val,
    unstring_df_val,
)

NSITES_COL_SPLITS = 20
_CONFIGS_COLLECTION = "gpw_test_configs"
_CONFIGSETS_COLLECTION = "gpw_test_config_sets"
_DATASETS_COLLECTION = "gpw_test_datasets"
_PROPOBJECT_COLLECTION = "gpw_test_prop_objects"
_MAX_STRING_LEN = 60000

# from kim_property.definition import PROPERTY_ID as VALID_KIM_ID

# from kim_property.definition import check_property_definition


def generate_string():
    return get_random_string(12, allowed_chars=string.ascii_lowercase + "1234567890")


class SparkDataLoader:
    def __init__(
        self,
        table_prefix: str = "ndb.colabfit.dev",
        endpoint=None,
        access_key=None,
        access_secret=None,
        check_ids_batch_size=500,
    ):
        self.table_prefix = table_prefix
        self.spark = SparkSession.builder.appName("ColabfitDataLoader").getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
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
        self.prop_object_table = f"{self.table_prefix}.{_PROPOBJECT_COLLECTION}"
        self.check_unique_ids_batch_size = check_ids_batch_size

    def get_vastdb_session(self, endpoint, access_key: str, access_secret: str):
        return Session(endpoint=endpoint, access=access_key, secret=access_secret)

    def set_vastdb_session(self, endpoint, access_key: str, access_secret: str):
        self.session = self.get_vastdb_session(endpoint, access_key, access_secret)

    def add_elem_to_col(df, col_name: str, elem: str):
        df_added_elem = df.withColumn(
            col_name,
            sf.when(
                sf.col(col_name).isNull(), sf.array().cast(ArrayType(StringType()))
            ).otherwise(sf.col(col_name)),
        )
        df_added_elem = df_added_elem.withColumn(
            col_name, sf.array_union(sf.col(col_name), sf.array(sf.lit(elem)))
        )
        return df_added_elem

    def delete_from_table(self, table_name: str, ids: list[str]):
        self.spark.sql(f"delete from {table_name} where id in {tuple(ids)}")

    def check_unique_ids(self, table_name: str, ids: list[str]):
        if not self.spark.catalog.tableExists(table_name):
            print(f"Table {table_name} does not yet exist.")
            return True
        batched_ids = batched(ids, self.check_unique_ids_batch_size)
        for i, batch in tqdm(
            enumerate(batched_ids),
            desc=f"Checking for duplicate ids in {table_name.split('.')[-1]}",
        ):
            broadcast_ids = self.spark.sparkContext.broadcast(batch)
            df = self.spark.read.table(table_name)
            dupes_exist = df.filter(sf.col("id").isin(broadcast_ids.value)).limit(1)
            if len(dupes_exist.collect()) > 0:
                print(f"Duplicate IDs found in table {table_name}")
                return False
        return True

    def write_table(
        self,
        spark_df,
        table_name: str,
        ids_filter: list[str] = None,
        check_length_col: str = None,
    ):
        """Include self.table_prefix in the table name when passed to this function"""
        string_cols = [
            f.name for f in spark_df.schema if f.dataType.typeName() == "array"
        ]
        string_col_udf = sf.udf(stringify_df_val, StringType())
        for col in string_cols:
            spark_df = spark_df.withColumn(col, string_col_udf(sf.col(col)))
        if ids_filter is not None:
            spark_df = spark_df.filter(sf.col("id").isin(ids_filter))
        print("length of df to write to table: ", spark_df.count())
        ids = [x["id"] for x in spark_df.select("id").collect()]
        all_unique = self.check_unique_ids(table_name, ids)
        if all_unique:
            spark_df.write.mode("append").saveAsTable(table_name)
        else:
            print("Duplicate IDs found in table. Not writing.")
            return False

    def find_existing_co_rows_append_elem(
        self,
        co_df,
        cols: list[str],
        elems: list[str],
    ):
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(elems, str):
            elems = [elems]
        col_types = {
            "id": StringType(),
            "last_modified": TimestampType(),
            "$row_id": IntegerType(),
        }
        arr_cols = []
        for col in cols:
            col_types[col] = get_spark_field_type(config_schema, col)
            is_arr = get_spark_field_type(config_df_schema, col)
            if is_arr.typeName() == "array":
                arr_cols.append(col)
        update_cols = [col for col in col_types if col not in ["id", "$row_id"]]
        # query_schema = StructType(
        #     [
        #         StructField(col, col_types[col], False)
        #         for i, col in enumerate(update_cols + ["id", "$row_id"])
        #     ]
        # )
        total_write_cols = update_cols + ["$row_id"]
        ids = [x["id"] for x in co_df.select("id").collect()]
        batched_ids = batched(ids, 10000)
        new_ids = []
        existing_ids = []
        for id_batch in batched_ids:
            id_batch = list(set(id_batch))
            # We only have to use vastdb-sdk here bc we need the '$row_id' column
            with self.session.transaction() as tx:
                table_path = self.config_table.split(".")
                table = (
                    tx.bucket(table_path[1]).schema(table_path[2]).table(table_path[3])
                )
                rec_batch = table.select(
                    predicate=table["id"].isin(id_batch),
                    columns=update_cols + ["id"],
                    internal_row_id=True,
                )
                spark_schema = StructType(
                    [
                        StructField(col, col_types[col], True)
                        for i, col in enumerate(update_cols)
                    ]
                    + [
                        StructField("id", StringType(), False),
                        StructField("$row_id", IntegerType(), False),
                    ]
                )
                rec_batch = rec_batch.read_all()
                duplicate_df = self.spark.createDataFrame(
                    rec_batch.to_pylist(), schema=spark_schema
                )
                print(f"length of df: {duplicate_df.count()}")
            unstring_udf = sf.udf(unstring_df_val, ArrayType(StringType()))
            for col_name, col_type in col_types.items():
                if col_name in arr_cols:
                    duplicate_df = duplicate_df.withColumn(
                        col_name, unstring_udf(sf.col(col_name))
                    )
            for col, elem in zip(cols, elems):
                if col == "labels":
                    co_df_labels = co_df.select("id", "labels").collect()
                    duplicate_df = (
                        duplicate_df.withColumnRenamed("labels", "labels_dup")
                        .join(
                            co_df_labels.withColumnRenamed("labels", "labels_co_df"),
                            on="id",
                        )
                        .withColumn(
                            "labels",
                            sf.array_distinct(
                                sf.array_union("labels_dup", "labels_co_df")
                            ),
                        )
                        .drop("labels_dup", "labels_co_df")
                    )

                else:
                    duplicate_df = duplicate_df.withColumn(
                        col,
                        sf.array_distinct(
                            sf.array_union(sf.col(col), sf.array(sf.lit(elem)))
                        ),
                    )
            existing_ids_batch = [x["id"] for x in duplicate_df.select("id").collect()]
            new_ids_batch = [id for id in id_batch if id not in existing_ids_batch]
            string_udf = sf.udf(stringify_df_val, StringType())
            for col_name in duplicate_df.columns:
                if col_name in arr_cols:
                    duplicate_df = duplicate_df.withColumn(
                        col_name, string_udf(sf.col(col_name))
                    )
            update_time = dateutil.parser.parse(
                datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            )
            duplicate_df = duplicate_df.withColumn(
                "last_modified", sf.lit(update_time).cast("timestamp")
            )
            update_schema = StructType(
                [StructField(col, col_types[col], False) for col in total_write_cols]
            )
            arrow_schema = spark_schema_to_arrow_schema(update_schema)
            update_table = pa.table(
                [
                    pa.array(col)
                    for col in zip(*duplicate_df.select(total_write_cols).collect())
                ],
                schema=arrow_schema,
            )
            with self.session.transaction() as tx:
                table = (
                    tx.bucket(table_path[1]).schema(table_path[2]).table(table_path[3])
                )
                table.update(
                    rows=update_table,
                    columns=update_cols,
                )
            new_ids.extend(new_ids_batch)
            existing_ids.extend(existing_ids_batch)

        return (new_ids, list(set(existing_ids)))

    def find_existing_po_rows_append_elem(
        self,
        ids: list[str],
        po_df=None,
    ):
        schema = StructType(
            [
                StructField("id", StringType(), False),
                StructField("multiplicity", IntegerType(), True),
                StructField("last_modified", TimestampType(), False),
                StructField("$row_id", IntegerType(), False),
            ]
        )
        arrow_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("multiplicity", pa.int32()),
                pa.field("last_modified", pa.timestamp("ns")),
                pa.field("$row_id", pa.int32()),
            ]
        )
        batched_ids = batched(ids, 10000)
        new_ids = []
        existing_ids = []
        columns = ["id", "multiplicity", "last_modified"]
        po_update_df = po_df.select("id", "multiplicity").withColumnRenamed(
            "multiplicity", "multiplicity_update"
        )
        for id_batch in batched_ids:
            # broadcast_ids = self.spark.sparkContext.broadcast(id_batch)
            id_batch = list(set(id_batch))
            # We only have to use vastdb-sdk here bc we need the '$row_id' column
            with self.session.transaction() as tx:
                table_name = self.prop_object_table
                # string would be 'ndb.colabfit.dev.[table name]'
                table_path = table_name.split(".")
                table = (
                    tx.bucket(table_path[1]).schema(table_path[2]).table(table_path[3])
                )
                rec_batch = table.select(
                    predicate=table["id"].isin(id_batch),
                    columns=columns,
                    internal_row_id=True,
                )
                rec_batch = rec_batch.read_all()
                duplicate_df = self.spark.createDataFrame(
                    rec_batch.to_pylist(), schema=schema
                )
                print(f"length of df: {duplicate_df.count()}")

            update_time = dateutil.parser.parse(
                datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            )
            duplicate_df = (
                duplicate_df.join(po_update_df, on="id")
                .withColumn(
                    "multiplicity",
                    sf.col("multiplicity") + sf.col("multiplicity_update"),
                )
                .drop("multiplicity_update")
            )
            duplicate_df = duplicate_df.withColumn(
                "last_modified", sf.lit(update_time).cast("timestamp")
            )
            existing_ids_batch = [x["id"] for x in duplicate_df.select("id").collect()]
            new_ids_batch = [id for id in id_batch if id not in existing_ids_batch]
            update_table = pa.table(
                [pa.array(col) for col in zip(*duplicate_df.collect())],
                schema=arrow_schema,
            )
            with self.session.transaction() as tx:
                table = (
                    tx.bucket(table_path[1]).schema(table_path[2]).table(table_path[3])
                )
                table.update(
                    rows=update_table, columns=["multiplicity", "last_modified"]
                )
            new_ids.extend(new_ids_batch)
            existing_ids.extend(existing_ids_batch)

        return (new_ids, list(set(existing_ids)))

    def read_table(self, table_name: str, unstring: bool = False):
        """
        Include self.table_prefix in the table name when passed to this function.
        Ex: loader.read_table(loader.config_table, unstring=True)
        Arguments:
            table_name {str} -- Name of the table to read from database
        Keyword Arguments:
            unstring {bool} -- Convert stringified lists to lists (default: {False})
        Returns:
            DataFrame -- Spark DataFrame
        """
        unstring_schema_dict = {
            self.config_table: config_df_schema,
            self.config_set_table: configuration_set_df_schema,
            self.dataset_table: dataset_df_schema,
            self.prop_object_table: property_object_df_schema,
        }
        string_schema_dict = {
            self.config_table: config_schema,
            self.config_set_table: configuration_set_schema,
            self.dataset_table: dataset_schema,
            self.prop_object_table: property_object_schema,
        }
        df = self.spark.read.table(table_name)
        if unstring:
            schema = unstring_schema_dict[table_name]
            schema_type_dict = {f.name: f.dataType for f in schema}
            string_cols = [f.name for f in schema if f.dataType.typeName() == "array"]
            for col in string_cols:
                string_col_udf = sf.udf(unstring_df_val, schema_type_dict[col])
                df = df.withColumn(col, string_col_udf(sf.col(col)))
        else:
            schema = string_schema_dict[table_name]
        mismatched_cols = [
            x
            for x in [(f.name, f.dataType.typeName()) for f in df.schema]
            if x not in [(f.name, f.dataType.typeName()) for f in schema]
        ]
        if len(mismatched_cols) == 0:
            return df
        else:
            raise ValueError(
                f"Schema mismatch for table {table_name}. "
                f"Mismatched column types in DataFrame: {mismatched_cols}"
            )

    def zero_multiplicity(self, dataset_id):
        """Use to return multiplicity of POs for a given dataset to zero"""
        ids = (
            self.spark.read.table(self.prop_object_table)
            .filter(sf.col("dataset_id") == dataset_id)
            .select("id")
            .collect()
        )
        ids = [x["id"] for x in ids]
        batched_ids = batched(ids, 1000)
        for id_batch in batched_ids:
            id_batch = list(set(id_batch))
            with self.session.transaction() as tx:
                table_name = self.prop_object_table
                table_path = table_name.split(".")
                table = (
                    tx.bucket(table_path[1]).schema(table_path[2]).table(table_path[3])
                )
                rec_batch = table.select(
                    predicate=table["id"].isin(id_batch),
                    columns=["id", "multiplicity", "last_modified"],
                    internal_row_id=True,
                )
                spark_schema = StructType(
                    [
                        StructField("id", StringType(), False),
                        StructField("multiplicity", IntegerType(), True),
                        StructField("last_modified", TimestampType(), False),
                        StructField("$row_id", IntegerType(), False),
                    ]
                )
                rec_batch = rec_batch.read_all()
                df = self.spark.createDataFrame(
                    rec_batch.to_pylist(), schema=spark_schema
                )
            print(f"length of df: {df.count()}")
            df = df.withColumn("multiplicity", sf.lit(0))
            update_time = dateutil.parser.parse(
                datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            )
            df = df.withColumn("last_modified", sf.lit(update_time).cast("timestamp"))
            arrow_schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("multiplicity", pa.int32()),
                    pa.field("last_modified", pa.timestamp("ns")),
                    pa.field("$row_id", pa.int32()),
                ]
            )
            update_table = pa.table(
                [pa.array(col) for col in zip(*df.collect())], schema=arrow_schema
            )
            with self.session.transaction() as tx:
                table_name = self.prop_object_table
                table_path = table_name.split(".")
                table = (
                    tx.bucket(table_path[1]).schema(table_path[2]).table(table_path[3])
                )
                table.update(
                    rows=update_table,
                    columns=["multiplicity", "last_modified"],
                )

    def get_pos_cos_by_filter(
        self,
        po_filter_conditions: list[tuple[str, str, str | int | float | list]],
        co_filter_conditions: list[
            tuple[str, str, str | int | float | list | None]
        ] = None,
    ):
        """
        example filter conditions:
        po_filter_conditions = [("dataset_id", "=", "ds_id1"),
                                ("method", "like", "DFT%")]
        co_filter_conditions = [("nsites", ">", 15),
                                ('labels', 'array_contains', 'label1')]
        """
        po_df = self.read_table(
            self.prop_object_table, unstring=True
        ).withColumnRenamed("id", "po_id")
        po_df = po_df.withColumn(
            "configuration_id", sf.explode(sf.col("configuration_ids"))
        ).drop("configuration_ids")
        co_df = self.read_table(self.config_table, unstring=True).withColumnRenamed(
            "id", "co_id"
        )
        po_df = self.get_filtered_table(po_df, po_filter_conditions)
        if co_filter_conditions is not None:
            co_df = self.get_filtered_table(co_df, co_filter_conditions)
        co_po_df = co_df.join(
            po_df, co_df["co_id"] == po_df["configuration_id"], "inner"
        )
        return co_po_df

    def get_filtered_table(
        self,
        df: DataFrame,
        filter_conditions: list[tuple[str, str, str | int | float | list]],
    ):
        for i, (column, operand, condition) in enumerate(filter_conditions):
            if operand == "in":
                df = df.filter(sf.col(column).isin(condition))
            elif operand == "like":
                df = df.filter(sf.col(column).like(condition))
            elif operand == "rlike":
                df = df.filter(sf.col(column).rlike(condition))
            elif operand == "==":
                df = df.filter(sf.col(column) == condition)
            elif operand == "array_contains":
                df = df.filter(sf.array_contains(sf.col(column), condition))
            elif operand == ">":
                df = df.filter(sf.col(column) > condition)
            elif operand == "<":
                df = df.filter(sf.col(column) < condition)
            elif operand == ">=":
                df = df.filter(sf.col(column) >= condition)
            elif operand == "<=":
                df = df.filter(sf.col(column) <= condition)
            else:
                raise ValueError(
                    f"Operand {operand} not implemented in get_pos_cos_filter"
                )
        return df

    def stop_spark(self):
        self.spark.stop()


class PGDataLoader:
    """
    Class to load data from files to ColabFit PostgreSQL database
    """

    def __init__(
        self,
        appname="colabfit",
        url="jdbc:postgresql://localhost:5432/colabfit",
        database_name: str = None,
        env="./.env",
        table_prefix: str = None,
    ):
        # self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        JARFILE = os.environ.get("CLASSPATH")
        self.spark = (
            SparkSession.builder.appName(appname)
            .config("spark.jars", JARFILE)
            .getOrCreate()
        )

        user = os.environ.get("PGS_USER")
        password = os.environ.get("PGS_PASS")
        driver = os.environ.get("PGS_DRIVER")
        self.properties = {
            "user": user,
            "password": password,
            "driver": driver,
        }
        self.url = url
        self.database_name = database_name
        self.table_prefix = table_prefix
        findspark.init()

        self.format = "jdbc"  # for postgres local
        load_dotenv(env)
        self.config_table = _CONFIGS_COLLECTION
        self.config_set_table = _CONFIGSETS_COLLECTION
        self.dataset_table = _DATASETS_COLLECTION
        self.prop_object_table = _PROPOBJECT_COLLECTION

    def read_table(
        self,
    ):
        pass

    def get_spark(self):
        return self.spark

    def get_spark_context(self):
        return self.spark.sparkContext

    def write_table(self, spark_rows: list[dict], table_name: str, schema: StructType):
        df = self.spark.createDataFrame(spark_rows, schema=schema)

        df.write.jdbc(
            url=self.url,
            table=table_name,
            mode="append",
            properties=self.properties,
        )

    def update_co_rows_cs_id(self, co_ids: list[str], cs_id: str):
        with psycopg.connect(
            """dbname=colabfit user=%s password=%s host=localhost port=5432"""
            % (
                self.user,
                self.password,
            )
        ) as conn:
            cur = conn.execute(
                """UPDATE configurations
                        SET configuration_set_ids = concat(%s::text, \
                rtrim(ltrim(replace(configuration_set_ids,%s,''), '['),']'), %s::text)
                """,
                (
                    "[",
                    f", {cs_id}",
                    f", {cs_id}]",
                ),
                # WHERE id = ANY(%s)""",
                # (cs_id, co_ids),
            )
            conn.commit()


def batched(configs, n):
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
        nprocs: int = 2,
        configs: list[AtomicConfiguration] = None,
        prop_defs: list[dict] = None,
        prop_map: dict = None,
        dataset_id=None,
        energy_conjugate=None,
    ):
        self.configs = configs
        if isinstance(prop_defs, dict):
            prop_defs = [prop_defs]
        self.prop_defs = prop_defs
        self.prop_map = prop_map
        self.nprocs = nprocs
        self.dataset_id = dataset_id
        if self.dataset_id is None:
            self.dataset_id = self.generate_ds_id()
        print("Dataset ID:", self.dataset_id)
        if (
            any(["energy" in x for x in self.prop_defs])
            and "atomic_forces_pd" in self.prop_defs
        ):
            if energy_conjugate is None:
                raise warnings.warn(
                    "No energy conjugate column provided for atomic forces.\n"
                    'Please set with i.e. energy_conjugate="electronic_free_energy" '
                    "when initializing DataManager\nChoices: 'potential_energy', "
                    "'electronic_free_energy'"
                )
            else:
                self.energy_conjugate = energy_conjugate

    @staticmethod
    def _gather_co_po_rows(
        prop_defs: list[dict],
        prop_map: dict,
        dataset_id,
        configs: list[AtomicConfiguration],
    ):
        """Convert COs and DOs to Spark rows."""
        co_po_rows = []
        for config in configs:
            config.set_dataset_id(dataset_id)
            property = Property.from_definition(
                definitions=prop_defs,
                configuration=config,
                property_map=prop_map,
            )
            co_po_rows.append(
                (
                    config.spark_row,
                    property.spark_row,
                )
            )
        return co_po_rows

    def gather_co_po_rows_pool(
        self, config_chunks: list[list[AtomicConfiguration]], pool: multiprocessing.Pool
    ):
        """
        Wrapper for _gather_co_po_rows.
        Convert COs and DOs to Spark rows using multiprocessing Pool.
        Returns a batch of tuples of (configuration_row, property_row).
        """

        part_gather = partial(
            self._gather_co_po_rows, self.prop_defs, self.prop_map, self.dataset_id
        )
        return itertools.chain.from_iterable(pool.map(part_gather, list(config_chunks)))

    def gather_co_po_in_batches(self):
        """
        Wrapper function for gather_co_po_rows_pool.
        Yields batches of CO-DO rows, preventing configuration iterator from
        being consumed all at once.
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
        Wrapper function for gather_co_po_rows_pool.
        Yields batches of CO-DO rows, preventing configuration iterator from
        being consumed all at once.
        """
        chunk_size = 10000
        config_chunks = batched(self.configs, chunk_size)

        for chunk in config_chunks:
            yield list(
                self._gather_co_po_rows(
                    self.prop_defs, self.prop_map, self.dataset_id, chunk
                )
            )

    def load_co_po_to_vastdb(self, loader):
        if loader.spark.catalog.tableExists(loader.config_table):
            pos_with_mult = (
                loader.read_table(loader.prop_object_table)
                .filter(sf.col("dataset_id") == self.dataset_id)
                .filter(sf.col("multiplicity") > 0)
                .limit(1)
            )
            if pos_with_mult.count() > 0:
                print(
                    f"POs for dataset with ID {self.dataset_id} already exist in "
                    "database with multiplicity > 0.\nTo continue, set multiplicities "
                    f'to 0 with loader.zero_multiplicity("{self.dataset_id}")'
                )
                return
        if loader.spark.catalog.tableExists(loader.dataset_table):
            dataset_exists = loader.read_table(loader.dataset_table).filter(
                sf.col("id") == self.dataset_id
            )
            if dataset_exists.count() > 0:
                print(f"Dataset with ID {self.dataset_id} already exists in database.")

                return
        co_po_rows = self.gather_co_po_in_batches_no_pool()
        for co_po_batch in tqdm(
            co_po_rows,
            desc="Loading data to database: ",
            unit="batch",
        ):
            co_rows, po_rows = list(zip(*co_po_batch))
            print(f"\nNum co_rows in batch at load_co_po_to_vastdb: {len(co_rows)}")
            print(f"\nNum po_rows in batch at load_co_po_to_vastdb: {len(po_rows)}")
            if len(co_rows) == 0:
                continue
            else:
                pass
                co_df = loader.spark.createDataFrame(co_rows, schema=config_df_schema)
                po_df = loader.spark.createDataFrame(
                    po_rows, schema=property_object_df_schema
                )
                co_ids = [x["id"] for x in co_df.select("id").collect()]
                if len(set(co_ids)) < len(co_ids):
                    print(f"{len(co_ids) -len(set(co_ids))} duplicates found in CO RDD")
                    co_df = co_df.dropDuplicates(["id"])
                po_ids = [x["id"] for x in po_df.select("id").collect()]
                if len(set(po_ids)) < len(po_ids):
                    print(
                        f"{len(po_ids) - len(set(po_ids))} duplicates found in PO RDD"
                    )
                    multiplicity = po_df.groupBy("id").agg(sf.count("*").alias("count"))
                    po_df = po_df.dropDuplicates(["id"])
                    po_df = (
                        po_df.join(multiplicity, on="id", how="inner")
                        .withColumn("multiplicity", sf.col("count"))
                        .drop("count")
                    )
                co_ids = set(co_ids)

                all_unique_co = loader.check_unique_ids(loader.config_table, co_ids)
                all_unique_po = loader.check_unique_ids(
                    loader.prop_object_table, po_ids
                )
                if not all_unique_co:
                    print("updating old rows")
                    new_co_ids, update_co_ids = (
                        loader.find_existing_co_rows_append_elem(
                            co_df=co_df,
                            cols=["dataset_ids"],
                            elems=[self.dataset_id],
                        )
                    )
                    print(f"Config ids in batch: {len(update_co_ids)}")
                    if len(new_co_ids) > 0:
                        print(f"Writing {len(new_co_ids)} new rows to table")
                        loader.write_table(
                            co_df,
                            loader.config_table,
                            ids_filter=new_co_ids,
                            check_length_col="positions_00",
                        )
                else:
                    loader.write_table(
                        co_df,
                        loader.config_table,
                        check_length_col="positions_00",
                    )
                    print(f"Inserted {len(co_rows)} rows into {loader.config_table}")

                if not all_unique_po:
                    new_po_ids, update_po_ids = (
                        loader.find_existing_po_rows_append_elem(
                            po_df=po_df,
                            ids=po_ids,
                        )
                    )
                    print(
                        f"Updated {len(update_po_ids)} rows in "
                        f"{loader.prop_object_table}"
                    )
                    if len(new_po_ids) > 0:
                        # if self.energy_conjugate is not None:
                        #     po_df = po_df.withColumn(
                        #         "energy_conjugate_with_forces",
                        #         sf.col(self.energy_conjugate),
                        #     )
                        #     po_df = po_df.withColumn(
                        #         "energy_conjugate_with_forces_units", sf.lit("eV")
                        #     )
                        #     po_df = po_df.withColumn(
                        #         "energy_conjugate_with_forces_column",
                        #         sf.lit(self.energy_conjugate),
                        #     )
                        # Add the conjugate function here
                        loader.write_table(
                            po_df,
                            loader.prop_object_table,
                            ids_filter=new_po_ids,
                            check_length_col="atomic_forces_00",
                        )
                    print(
                        f"Inserted {len(new_po_ids)} rows into "
                        f"{loader.prop_object_table}"
                    )
                else:
                    # # Add the conjugate function here
                    # if self.energy_conjugate is not None:
                    #     po_df = po_df.withColumn(
                    #         "energy_conjugate_with_forces",
                    #         sf.col(self.energy_conjugate),
                    #     )
                    #     po_df = po_df.withColumn(
                    #         "energy_conjugate_with_forces_units", sf.lit("eV")
                    #     )
                    #     po_df = po_df.withColumn(
                    #         "energy_conjugate_with_forces_column",
                    #         sf.lit(self.energy_conjugate),
                    #     )
                    loader.write_table(
                        po_df,
                        loader.prop_object_table,
                        check_length_col="atomic_forces_00",
                    )
                    print(
                        f"Inserted {len(po_rows)} rows into {loader.prop_object_table}"
                    )

    def load_data_to_pg_in_batches(self, loader):
        """Load data to PostgreSQL in batches."""
        co_po_rows = self.gather_co_po_in_batches()

        for co_po_batch in tqdm(
            co_po_rows,
            desc="Loading data to database: ",
            unit="batch",
        ):
            co_rows, po_rows = list(zip(*co_po_batch))
            if len(co_rows) == 0:
                continue
            else:
                loader.write_table(
                    co_rows,
                    loader.config_table,
                    config_schema,
                )
                loader.write_table(
                    po_rows,
                    loader.prop_object_table,
                    property_object_schema,
                )

    def create_configuration_sets(
        self,
        loader,
        name_label_match: list[tuple],
    ):
        """
        Args for name_label_match in order:
        1. Regex pattern for matching CONFIGURATION NAMES
        2. Regex pattern for matching CONFIGURATION LABELS
        3. Name for configuration set
        4. Description for configuration set
        """
        config_set_rows = []
        config_df = loader.read_table(table_name=loader.config_table, unstring=True)
        config_df = config_df.filter(
            sf.array_contains(sf.col("dataset_ids"), self.dataset_id)
        )
        prop_df = loader.read_table(loader.prop_object_table, unstring=True)
        prop_df = prop_df.filter(sf.col("dataset_id") == self.dataset_id)
        for i, (names_match, label_match, cs_name, cs_desc) in tqdm(
            enumerate(name_label_match), desc="Creating Configuration Sets"
        ):
            print(
                f"names match: {names_match}, label: {label_match}, "
                f"cs_name: {cs_name}, cs_desc: {cs_desc}"
            )
            if names_match:
                config_set_query = config_df.withColumn(
                    "names_exploded", sf.explode(sf.col("names"))
                ).filter(sf.col("names_exploded").rlike(names_match))
            # Currently an AND operation on labels: labels col contains x AND y
            if label_match is not None:
                if isinstance(label_match, str):
                    label_match = [label_match]
                for label in label_match:
                    config_set_query = config_set_query.filter(
                        sf.array_contains(sf.col("labels"), label)
                    )
            prop_df = loader.read_table(loader.prop_object_table, unstring=True)
            prop_df = (
                prop_df.filter(sf.col("dataset_id") == self.dataset_id)
                .select("configuration_id", "multiplicity")
                .withColumnRenamed("configuration_id", "id")
            )
            config_set_query = config_set_query.join(prop_df, on="id", how="inner")
            t = time()
            config_set = ConfigurationSet(
                name=cs_name,
                description=cs_desc,
                config_df=config_set_query,
                dataset_id=self.dataset_id,
            )

            co_ids = [
                x["id"] for x in config_set_query.select("id").distinct().collect()
            ]
            print(f"Num config ids in config set: {len(co_ids)}")

            loader.find_existing_co_rows_append_elem(
                co_df=config_set_query,
                cols=["configuration_set_ids"],
                elems=config_set.spark_row["id"],
            )
            t_end = time() - t
            print(f"Time to create CS and update COs with CS-ID: {t_end}")

            config_set_rows.append(config_set.spark_row)
        config_set_df = loader.spark.createDataFrame(
            config_set_rows, schema=configuration_set_schema
        )
        loader.write_table(config_set_df, loader.config_set_table)
        return config_set_rows

    def create_dataset(
        self,
        loader,
        name: str,
        authors: list[str],
        publication_link: str,
        data_link: str,
        description: str,
        other_links: list[str] = None,
        dataset_id: str = None,
        labels: list[str] = None,
        data_license: str = "CC-BY-ND-4.0",
    ):
        if loader.spark.catalog.tableExists(loader.config_set_table):
            cs_ids = loader.read_table(loader.config_set_table).select("id").collect()
            if len(cs_ids) == 0:
                cs_ids = None
            else:
                cs_ids = [x["id"] for x in cs_ids]
        else:
            cs_ids = None
        config_df = loader.read_table(loader.config_table, unstring=True)
        config_df = config_df.filter(
            sf.array_contains(sf.col("dataset_ids"), dataset_id)
        )
        prop_df = loader.read_table(loader.prop_object_table, unstring=True)
        prop_df = prop_df.filter(sf.col("dataset_id") == dataset_id)
        ds = Dataset(
            name=name,
            authors=authors,
            config_df=config_df,
            prop_df=prop_df,
            publication_link=publication_link,
            data_link=data_link,
            description=description,
            other_links=other_links,
            dataset_id=dataset_id,
            labels=labels,
            data_license=data_license,
            configuration_set_ids=cs_ids,
        )
        ds_df = loader.spark.createDataFrame([ds.spark_row], schema=dataset_schema)
        loader.write_table(ds_df, loader.dataset_table)

    @staticmethod
    def generate_ds_id():
        # Maybe check to see whether the DS ID already exists?
        ds_id = ID_FORMAT_STRING.format("DS", generate_string(), 0)
        print("Generated new DS ID:", ds_id)
        return ds_id
