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
from django.utils.crypto import get_random_string
from dotenv import load_dotenv
from kim_property.definition import PROPERTY_ID as VALID_KIM_ID

# from kim_property.definition import check_property_definition
from pyspark.sql import DataFrame, SparkSession
from tqdm import tqdm
from unidecode import unidecode
from pyspark.sql.types import StructType

from colabfit import (  # ATOMS_NAME_FIELD,; EXTENDED_ID_STRING_NAME,;
    # MAX_STRING_LENGTH,; SHORT_ID_STRING_NAME,
    _CONFIGS_COLLECTION,
    _CONFIGSETS_COLLECTION,
    _DATASETS_COLLECTION,
    _PROPDEFS_COLLECTION,
    _PROPOBJECT_COLLECTION,
    ID_FORMAT_STRING,
)
from colabfit.tools.configuration import AtomicConfiguration, config_schema
from colabfit.tools.property import Property, property_object_schema

# from colabfit.tools.dataset import Dataset


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
        ds_id=None,
        *args,
        **kwargs,
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
        self.ds_id = ds_id
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
            config.dataset_id = dataset_id
            co_po_rows.append(
                (
                    config.spark_row.update({"dataset_ids": [dataset_id]}),
                    Property.from_definition(
                        prop_defs,
                        configuration=config,
                        property_map=prop_map,
                        dataset_id=dataset_id,
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
        # print("number of chunks", len(config_chunks))
        # print(len(config_chunks[0]))
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
                    _CONFIGS_COLLECTION,
                    config_schema,
                )
                loader.write_table(
                    po_rows,
                    _PROPOBJECT_COLLECTION,
                    property_object_schema,
                )

    @staticmethod
    def generate_ds_id(ds_id=None):
        if ds_id is None:
            # Maybe check to see whether the DS ID already exists?
            ds_id = ID_FORMAT_STRING.format("DS", generate_string(), 0)
            print("Generated new DS ID:", ds_id)
        return ds_id
