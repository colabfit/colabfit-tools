import datetime
import itertools
import json
import multiprocessing
import os
import string
import warnings
from functools import partial
from itertools import islice
from multiprocessing import Pool
from time import time
from types import GeneratorType

import findspark
import psycopg
import pyarrow as pa
import pyspark.sql.functions as sf
from django.utils.crypto import get_random_string
from dotenv import load_dotenv
from ibis import _
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from tqdm import tqdm
from unidecode import unidecode
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
    add_elem_to_row_dict,
    append_ith_element_to_rdd,
    arrow_record_batch_to_rdd,
    get_spark_field_type,
    spark_schema_to_arrow_schema,
    stringify_lists,
    stringify_row_dict,
    unstringify,
    unstringify_row_dict,
)

_CONFIGS_COLLECTION = "gpw_test_configs"
_CONFIGSETS_COLLECTION = "gpw_test_configsets"
_DATASETS_COLLECTION = "gpw_test_datasets"
_PROPOBJECT_COLLECTION = "gpw_test_propobjects"

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
    ):
        self.table_prefix = table_prefix
        self.spark = SparkSession.builder.appName("ColabfitDataLoader").getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.endpoint = endpoint
        self.access_key = access_key
        self.access_secret = access_secret
        self.config_table = f"{self.table_prefix}.{_CONFIGS_COLLECTION}"
        self.config_set_table = f"{self.table_prefix}.{_CONFIGSETS_COLLECTION}"
        self.dataset_table = f"{self.table_prefix}.{_DATASETS_COLLECTION}"
        self.prop_object_table = f"{self.table_prefix}.{_PROPOBJECT_COLLECTION}"

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
        batched_ids = batched(ids, 500)
        for i, batch in tqdm(
            enumerate(batched_ids),
            desc=f"Checking for duplicate ids in {table_name.split('.')[-1]}",
        ):
            batch = list(batch)
            broadcast_ids = self.spark.sparkContext.broadcast(batch)
            df = self.spark.read.table(table_name)
            dupes_exist = df.filter(sf.col("id").isin(broadcast_ids.value)).limit(1)
            if len(dupes_exist.collect()) > 0:
                print(f"Duplicate IDs found in table {table_name}")
                return False
        return True

    def write_table(
        self,
        spark_rdd,
        table_name: str,
        schema: StructType,
        ids_filter: list[str] = None,
    ):
        """Include self.table_prefix in the table name when passed to this function"""
        if ids_filter is not None:
            rdd = spark_rdd.map(stringify_lists).filter(lambda x: x["id"] in ids_filter)
        else:
            rdd = spark_rdd.map(stringify_lists)
        ids = rdd.map(lambda x: x["id"]).collect()
        all_unique = self.check_unique_ids(table_name, ids)
        if all_unique:
            rdd.toDF(schema).write.mode("append").saveAsTable(table_name)
        else:
            print("Duplicate IDs found in table. Not writing.")
            return False

    def reduce_po_rdd(self, po_rdd):
        po_co_ids = (
            po_rdd.map(lambda x: (x["id"], x["configuration_ids"][0]))
            .groupByKey()
            .mapValues(list)
        )
        po_id_map = po_co_ids.collectAsMap()
        broadcast_map = self.spark.sparkContext.broadcast(po_id_map)

        def replace_id_val(row):
            row["configuration_ids"] = broadcast_map.value[row["id"]]
            return row

        po_rdd = po_rdd.map(replace_id_val)
        po_rdd = (
            po_rdd.map(lambda x: (x["id"], x))
            .reduceByKey(lambda a, b: a)
            .map(lambda x: x[1])
        )
        return po_rdd

    def find_existing_rows_append_elem(
        self,
        table_name: str,
        ids: list[str],
        cols: list[str],
        elems: list[str],
        write_schema: StructType,
        po_rdd=None,
    ):
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(elems, str):
            elems = [elems]
        col_types = {"id": StringType(), "$row_id": IntegerType()}
        for col in cols:
            col_types[col] = get_spark_field_type(write_schema, col)
        update_cols = [col for col in col_types if col != "id"]
        query_schema = StructType(
            [
                StructField(col, col_types[col], False)
                for i, col in enumerate(cols + ["id", "$row_id"])
            ]
        )

        partial_batch_to_rdd = partial(arrow_record_batch_to_rdd, query_schema)
        batched_ids = batched(ids, 10000)
        new_ids = []
        existing_ids = []
        for id_batch in batched_ids:
            # broadcast_ids = self.spark.sparkContext.broadcast(id_batch)
            id_batch = list(set(id_batch))
            # We only have to use vastdb-sdk here bc we need the '$row_id' column
            with self.session.transaction() as tx:
                # string would be 'ndb.colabfit.dev.[table name]'
                table_path = table_name.split(".")
                table = (
                    tx.bucket(table_path[1]).schema(table_path[2]).table(table_path[3])
                )
                rec_batch = table.select(
                    predicate=table["id"].isin(id_batch),
                    columns=cols + ["id"],
                    internal_row_id=True,
                )
                rec_batch = rec_batch.read_all()
                duplicate_rdd = self.spark.sparkContext.parallelize(
                    list(partial_batch_to_rdd(rec_batch))
                )
                print(f"length of rdd: {duplicate_rdd.count()}")
            duplicate_rdd = duplicate_rdd.map(unstringify_row_dict)
            for col, elem in zip(cols, elems):
                # Add 'labels' to this?
                if col == "configuration_ids":
                    if po_rdd is None:
                        raise ValueError(
                            "Need to pass po_rdd when updating configuration_ids"
                        )
                    po_co_id_map = (
                        po_rdd.map(lambda x: (x["id"], x["configuration_ids"][0]))
                        .groupByKey()
                        .mapValues(list)
                        .collect()
                    )
                    po_co_id_map = dict(po_co_id_map)
                    co_ids = duplicate_rdd.map(
                        lambda x: po_co_id_map[x["id"]]
                    ).zipWithIndex()
                    duplicate_rdd = duplicate_rdd.zipWithIndex()
                    duplicate_rdd = duplicate_rdd.map(lambda x: (x[1], x[0]))
                    co_ids = co_ids.map(lambda x: (x[1], x[0]))
                    duplicate_rdd = duplicate_rdd.join(co_ids).map(
                        append_ith_element_to_rdd
                    )

                else:
                    partial_add = partial(add_elem_to_row_dict, col, elem)
                    duplicate_rdd = duplicate_rdd.map(partial_add)
            existing_ids_batch = duplicate_rdd.map(lambda x: x["id"]).collect()
            new_ids_batch = [id for id in id_batch if id not in existing_ids_batch]
            duplicate_rdd = duplicate_rdd.map(stringify_row_dict)
            rdd_collect = duplicate_rdd.map(
                lambda x: [x[col] for col in update_cols]
            ).collect()
            update_schema = StructType(
                [StructField(col, col_types[col], False) for col in update_cols]
            )
            arrow_schema = spark_schema_to_arrow_schema(update_schema)
            update_table = pa.table(
                [pa.array(col) for col in zip(*rdd_collect)], schema=arrow_schema
            )
            with self.session.transaction() as tx:
                table = (
                    tx.bucket(table_path[1]).schema(table_path[2]).table(table_path[3])
                )
                table.update(rows=update_table)
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
        schema_dict = {
            self.config_table: config_df_schema,
            self.config_set_table: configuration_set_df_schema,
            self.dataset_table: dataset_df_schema,
            self.prop_object_table: property_object_df_schema,
        }
        if unstring:
            df = self.spark.read.table(table_name)
            return df.rdd.map(unstringify).toDF(schema_dict[table_name])
        else:
            return self.spark.read.table(table_name)

    def get_pos_cos_by_filter(
        self,
        po_filter_conditions: list[tuple[str, str, str | int | float | list]],
        co_filter_conditions: list[
            tuple[str, str, str | int | float | list | None]
        ] = None,
    ):
        """
        example filter conditions:
        po_filter_conditions = [("dataset_ids", "in", ["po_id1", "po_id2"]),
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
                        SET configuration_set_ids = concat(%s::text, rtrim(ltrim(replace(configuration_set_ids,%s,''), '['),']'), %s::text)""",
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
    ):
        self.configs = configs
        if isinstance(prop_defs, dict):
            prop_defs = [prop_defs]
        self.prop_defs = prop_defs
        self.prop_map = prop_map
        self.nprocs = nprocs
        self.dataset_id = dataset_id
        print("Dataset ID:", self.dataset_id)
        if self.dataset_id is None:
            self.dataset_id = self.generate_ds_id()

    # TODO: consider moving where we assign dataset_id to configs
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
                prop_defs,
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

        # For running without multiprocessing on notebook
        # part_gather = partial(
        #     self._gather_co_po_rows,
        #     self.prop_defs,
        #     self.prop_map,
        # )
        # while batch := tuple(islice(part_gather(self.configs), chunk_size)):
        #     yield batch
        #     break

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
                co_rdd = loader.spark.sparkContext.parallelize(co_rows)
                po_rdd = loader.spark.sparkContext.parallelize(po_rows)
                co_ids = co_rdd.map(lambda x: x["id"]).collect()
                if len(set(co_ids)) < len(co_ids):
                    print(f"{len(co_ids) -len(set(co_ids))} duplicates found in CO RDD")
                    co_rdd = (
                        co_rdd.map(lambda x: (x["id"], x))
                        .reduceByKey(lambda a, b: a)
                        .map(lambda x: x[1])
                    )
                    print(f"New length of co_rdd: {co_rdd.count()}")
                po_ids = po_rdd.map(lambda x: x["id"]).collect()
                if len(set(po_ids)) < len(po_ids):
                    print(
                        f"{len(po_ids) - len(set(po_ids))} duplicates found in PO RDD"
                    )
                    po_rdd = loader.reduce_po_rdd(po_rdd)
                all_unique_co = loader.check_unique_ids(loader.config_table, co_ids)
                all_unique_po = loader.check_unique_ids(
                    loader.prop_object_table, po_ids
                )
                if not all_unique_co:
                    print("updating old rows")
                    new_co_ids, update_co_ids = loader.find_existing_rows_append_elem(
                        table_name=loader.config_table,
                        ids=co_ids,
                        cols=["dataset_ids"],
                        elems=[self.dataset_id],
                        write_schema=config_schema,
                    )
                    print(f"Config ids in batch: {len(update_co_ids)}")
                    print("writing new rows after updating old rows")
                    loader.write_table(
                        co_rdd,
                        loader.config_table,
                        config_schema,
                        ids_filter=new_co_ids,
                    )
                else:
                    loader.write_table(
                        co_rdd,
                        loader.config_table,
                        config_schema,
                    )
                    print(f"Inserted {len(co_rows)} rows into {loader.config_table}")

                if not all_unique_po:
                    new_po_ids, update_po_ids = loader.find_existing_rows_append_elem(
                        po_rdd=po_rdd,
                        table_name=loader.prop_object_table,
                        ids=po_ids,
                        cols=["dataset_ids", "configuration_ids"],
                        elems=[self.dataset_id, "placeholder"],
                        write_schema=property_object_schema,
                    )
                    print(f"length of update_po_ids {len(update_po_ids)}")
                    print(
                        f"length of set update_po_ids {len(list(set(update_po_ids)))}"
                    )
                    print(
                        f"Updated {len(update_po_ids)} rows in {loader.prop_object_table}"
                    )
                    loader.write_table(
                        po_rdd,
                        loader.prop_object_table,
                        property_object_schema,
                        ids_filter=new_po_ids,
                    )
                    print(
                        f"Inserted {len(new_po_ids)} rows into {loader.prop_object_table}"
                    )
                else:
                    loader.write_table(
                        po_rdd,
                        loader.prop_object_table,
                        property_object_schema,
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
        # below args in order:
        # [config-name-regex-pattern], [config-label-regex-pattern], \
        # [config-set-name], [config-set-description]
        name_label_match: list[tuple],
    ):
        config_set_rows = []
        config_df = loader.read_table(table_name=loader.config_table, unstring=True)
        config_df = config_df.filter(
            sf.array_contains(sf.col("dataset_ids"), self.dataset_id)
        )
        for i, (names_match, label_match, cs_name, cs_desc) in tqdm(
            enumerate(name_label_match), desc="Creating Configuration Sets"
        ):
            print(
                f"names match: {names_match}, label {label_match}, "
                f"cs_name {cs_name}, cs_desc {cs_desc}"
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
            t = time()
            config_set = ConfigurationSet(
                name=cs_name,
                description=cs_desc,
                config_df=config_set_query,
                dataset_id=self.dataset_id,
            )
            t_end = time() - t
            print(f"Time to create config set: {t_end}")
            co_ids = [
                x["id"] for x in config_set_query.select("id").distinct().collect()
            ]
            print(f"Num config ids in config set: {len(co_ids)}")
            t = time()
            loader.find_existing_rows_append_elem(
                table_name=loader.config_table,
                ids=co_ids,
                cols="configuration_set_ids",
                elems=config_set.spark_row["id"],
                write_schema=config_schema,
            )
            t_end = time() - t
            print(f"Time to update co-ids: {t_end}")

            config_set_rows.append(config_set.spark_row)
        config_rdd = loader.spark.sparkContext.parallelize(config_set_rows)
        loader.write_table(
            config_rdd, loader.config_set_table, schema=configuration_set_schema
        )
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
        cs_ids = loader.read_table(loader.config_set_table).select("id").collect()
        if len(cs_ids) == 0:
            cs_ids = None
        else:
            cs_ids = [x["id"] for x in cs_ids]
        config_df = loader.read_table(loader.config_table, unstring=True)
        config_df = config_df.filter(
            sf.array_contains(sf.col("dataset_ids"), dataset_id)
        )
        prop_df = loader.read_table(loader.prop_object_table, unstring=True)
        prop_df = prop_df.filter(sf.array_contains(sf.col("dataset_ids"), dataset_id))
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
        ds_rdd = loader.spark.sparkContext.parallelize([ds.spark_row])
        loader.write_table(ds_rdd, loader.dataset_table, schema=dataset_schema)

    @staticmethod
    def generate_ds_id():
        # Maybe check to see whether the DS ID already exists?
        ds_id = ID_FORMAT_STRING.format("DS", generate_string(), 0)
        print("Generated new DS ID:", ds_id)
        return ds_id
