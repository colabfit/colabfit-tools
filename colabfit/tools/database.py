import datetime
import itertools
import json
import multiprocessing
import os
import string
import time
import warnings
from functools import partial
from itertools import islice
from multiprocessing import Pool
from types import GeneratorType

import findspark
import pyspark.sql.functions as sf
from pyspark.sql.types import ArrayType, StringType
from django.utils.crypto import get_random_string
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from tqdm import tqdm
from unidecode import unidecode

from colabfit import (  # ATOMS_NAME_FIELD,; EXTENDED_ID_STRING_NAME,;; MAX_STRING_LENGTH,; SHORT_ID_STRING_NAME,
    _CONFIGS_COLLECTION,
    _CONFIGSETS_COLLECTION,
    _DATASETS_COLLECTION,
    _PROPDEFS_COLLECTION,
    _PROPOBJECT_COLLECTION,
    ID_FORMAT_STRING,
)
from colabfit.tools.configuration import AtomicConfiguration
from colabfit.tools.dataset import Dataset
from colabfit.tools.configuration_set import ConfigurationSet
from colabfit.tools.property import Property
from colabfit.tools.schema import (
    config_schema,
    dataset_schema,
    property_object_schema,
    configuration_set_schema,
)

# from kim_property.definition import PROPERTY_ID as VALID_KIM_ID

# from kim_property.definition import check_property_definition


def generate_string():
    return get_random_string(12, allowed_chars=string.ascii_lowercase + "1234567890")


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
        # Commented out below may not be necessary/may not work,
        # but may be used in cases like postgres, where a prefix like 'public.' is used
        # if table_prefix is not None:
        #     self.config_table = table_prefix + _CONFIGS_COLLECTION
        #     self.config_set_table = table_prefix + _CONFIGSETS_COLLECTION
        #     self.dataset_table = table_prefix + _DATASETS_COLLECTION
        #     self.prop_def_table = table_prefix + _PROPDEFS_COLLECTION
        #     self.prop_object_table = table_prefix + _PROPOBJECT_COLLECTION
        # else:
        self.config_table = _CONFIGS_COLLECTION
        self.config_set_table = _CONFIGSETS_COLLECTION
        self.dataset_table = _DATASETS_COLLECTION
        self.prop_def_table = _PROPDEFS_COLLECTION
        self.prop_object_table = _PROPOBJECT_COLLECTION

    def load_data(
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

    def update_rows(self, spark_rows: list[dict], table_name: str, schema: StructType):
        df = self.spark.createDataFrame(spark_rows, schema=schema)
        df.write.jdbc(
            url=self.url,
            table=table_name,
            mode="overwrite",
            properties=self.properties,
        )


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
    # TODO: properly manage multiple dataset-ids in case of hash/id collision
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
            co_po_rows.append(
                (
                    config.spark_row,
                    Property.from_definition(
                        prop_defs,
                        configuration=config,
                        property_map=prop_map,
                    ).spark_row,
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

    def load_data_to_pg_in_batches(self, loader: PGDataLoader):
        """Load data to PostgreSQL database in batches."""
        co_po_rows = self.gather_co_po_in_batches()

        for co_po_batch in tqdm(
            co_po_rows,
            desc="Loading data to PostgreSQL: ",
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

    def create_configuration_set(
        self,
        loader,
        name: str,
        description: str,
        name_label_match: list[tuple],
        dataset_id: str,
    ):
        config_df = (
            loader.spark.read.jdbc(
                url=loader.url, table=loader.config_table, properties=loader.properties
            )
            .withColumn(
                "names_unstrung", sf.from_json(sf.col("names"), ArrayType(StringType()))
            )
            .withColumn(
                "labels_unstrung",
                sf.from_json(sf.col("labels"), ArrayType(StringType())),
            )
            .withColumn(
                "dataset_ids_unstrung",
                sf.from_json("dataset_ids", ArrayType(StringType())),
            )
            .drop("names", "labels", "dataset_ids")
            .withColumnRenamed("names_unstrung", "names")
            .withColumnRenamed("labels_unstrung", "labels")
            .withColumnRenamed("dataset_ids_unstrung", "dataset_ids")
            .filter(sf.array_contains(sf.col("dataset_ids"), dataset_id))
        )

        for i, (names_match, label_match) in enumerate(name_label_match):
            if names_match:
                config_set_query = config_df.withColumn(
                    "names_exploded", sf.explode(sf.col("names"))
                ).filter(
                    sf.regexp_like(sf.col("names_exploded"), sf.lit(rf"{names_match}"))
                )
            if label_match:
                config_set_query = config_set_query.withColumn(
                    "labels_exploded", sf.explode(sf.col("labels"))
                ).filter(
                    sf.regexp_like(sf.col("labels_exploded"), sf.lit(rf"{label_match}"))
                )
            co_ids = [x[0] for x in config_set_query.select("id").distinct().collect()]
            loader.spark.write.jdbc(
                url=loader.url, table=loader.config_table, properties=loader.properties
            )
            row = config_set.spark_row
            loader.write_table(
                [row], loader.config_set_table, schema=configuration_set_schema
            )

    def create_write_dataset(
        self,
        loader,
        name: str,
        authors: list[str],
        publication_link: str,
        data_link: str,
        description: str,
        labels: list[str],
    ):
        if loader.table_prefix is not None:
            config_table = f"{loader.table_prefix}.{loader.config_table}"
            prop_table = f"{loader.table_prefix}.{loader.prop_table}"
        else:
            config_table = loader.config_table
            prop_table = loader.prop_object_table

        config_df = (
            loader.spark.read.jdbc(
                url=loader.url, table=config_table, properties=loader.properties
            )
            .withColumn(
                "ds_ids_unstrung",
                sf.from_json(sf.col("dataset_ids"), sf.ArrayType(sf.StringType())),
            )
            .filter(sf.array_contains("ds_ids_unstrung", self.dataset_id))
            .drop("ds_ids_unstrung")
        )
        prop_df = (
            loader.spark.read.jdbc(
                url=loader.url,
                table=prop_table,
                properties=loader.properties,
            )
            .withColumn(
                "ds_ids_unstrung",
                sf.from_json(sf.col("dataset_ids"), sf.ArrayType(sf.StringType())),
            )
            .filter(sf.array_contains("ds_ids_unstrung", self.dataset_id))
            .drop("ds_ids_unstrung")
        )
        dataset = Dataset(
            config_df=config_df,
            prop_df=prop_df,
            name=name,
            authors=authors,
            publication_link=publication_link,
            data_link=data_link,
            description=description,
            labels=labels,
            dataset_id=self.dataset_id,
        )
        row = dataset.spark_row
        loader.write_table([row], loader.dataset_table, schema=dataset_schema)

    @staticmethod
    def generate_ds_id():
        # Maybe check to see whether the DS ID already exists?
        ds_id = ID_FORMAT_STRING.format("DS", generate_string(), 0)
        print("Generated new DS ID:", ds_id)
        return ds_id
