import hashlib
import itertools
import json
import string
from functools import partial
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from time import time
from types import GeneratorType

import boto3
import psycopg
from ase import Atoms
from botocore.exceptions import ClientError
from psycopg import sql
from psycopg.rows import dict_row
from tqdm import tqdm

from colabfit import ID_FORMAT_STRING
from colabfit.tools.pg.configuration import AtomicConfiguration
from colabfit.tools.pg.configuration_set import ConfigurationSet
from colabfit.tools.pg.dataset import Dataset
from colabfit.tools.pg.property import Property
from colabfit.tools.pg.schema import (
    config_md_schema,
    config_schema,
    configuration_set_schema,
    dataset_schema,
    property_definition_schema,
    property_object_md_schema,
    property_object_schema,
)
from colabfit.tools.pg.utilities import get_last_modified

VAST_BUCKET_DIR = "colabfit-data"
VAST_METADATA_DIR = "data/MD"
NSITES_COL_SPLITS = 20


def generate_string():
    return "".join(secrets.choice(string.ascii_lowercase + "1234567890") for _ in range(12))


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
        dbname: str,
        user: str,
        port: int,
        host: str,
        password: str = None,
        nprocs: int = 1,
        standardize_energy: bool = False,
        read_write_batch_size: int = 10000,
    ):
        """
        Args:
            dbname (str): Name of the database.
            user (str): User name.
            port (int): Port number.
            host (str): Host name.
            password (str): Password.
            nprocs (int): Number of processes to use if using multiprocessing
                (i.e. while reading data files).
            standardize_energy (bool): Whether to standardize energy.
            read_write_batch_size (int): Batch size for reading and writing data.
        """
        self.dbname = dbname
        self.user = user
        self.port = port
        self.user = user
        self.password = password
        self.host = host
        self.read_write_batch_size = read_write_batch_size
        self.nprocs = nprocs
        self.standardize_energy = standardize_energy

    @staticmethod
    def _gather_co_po_rows(
        configs: list[AtomicConfiguration],
        prop_defs: list[dict],
        prop_map: dict,
        dataset_id,
        standardize_energy: bool = True,
    ):
        """Convert COs and DOs to Spark rows."""
        co_po_rows = []
        for config in configs:
            config.set_dataset_id(dataset_id)
            # TODO: Add PO schema as input to this method so to_row_dict works better
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

    def gather_co_po_rows_pool(
        self,
        config_chunks: list[list[AtomicConfiguration]],
        pool,
        dataset_id=None,
        prop_map=None,
    ):
        """
        Wrapper for _gather_co_po_rows.
        Convert COs and DOs to Spark rows using multiprocessing Pool.
        Returns a batch of tuples of (configuration_row, property_row).
        """

        if dataset_id is None:
            dataset_id = generate_ds_id()

        part_gather = partial(
            self._gather_co_po_rows,
            prop_defs=self.get_property_definitions(),
            prop_map=prop_map,
            dataset_id=dataset_id,
            standardize_energy=self.standardize_energy,
        )
        return itertools.chain.from_iterable(pool.map(part_gather, list(config_chunks)))

    def gather_co_po_in_batches(self, configs, dataset_id=None, prop_map=None):
        """
        Wrapper function for gather_co_po_rows_pool.
        Yields batches of CO-DO rows, preventing configuration iterator from
        being consumed all at once.
        """
        chunk_size = 1000
        config_chunks = batched(configs, chunk_size)
        with Pool(self.nprocs) as pool:
            while True:
                config_batches = list(islice(config_chunks, self.nprocs))
                if not config_batches:
                    break
                else:
                    yield list(
                        self.gather_co_po_rows_pool(
                            config_batches, pool, dataset_id, prop_map
                        )
                    )

    def load_data_loader_call(self, loader):
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

    @staticmethod
    def get_co_sql():
        columns = [x.name for x in config_md_schema.columns]
        sql_compose = sql.SQL(" ").join(
            [
                sql.SQL("INSERT INTO"),
                sql.Identifier(config_md_schema.name),
                sql.SQL("("),
                sql.SQL(",").join(map(sql.Identifier, columns)),
                sql.SQL(") VALUES ("),
                sql.SQL(",").join(sql.Placeholder() * len(columns)),
                sql.SQL(")"),
                sql.SQL("ON CONFLICT (hash) DO UPDATE SET"),
                sql.Identifier("dataset_ids"),
                sql.SQL("= array_append("),
                sql.Identifier("configurations.dataset_ids"),
                sql.SQL(","),
                sql.Placeholder(),
                sql.SQL(") ,"),
                sql.Identifier("names"),
                sql.SQL("= array_append("),
                sql.Identifier("configurations.names"),
                sql.SQL(","),
                sql.Placeholder(),
                sql.SQL(");"),
            ]
        )
        return sql_compose

    @staticmethod
    def get_po_sql():
        columns = [x.name for x in property_object_md_schema.columns]
        sql_compose = sql.SQL(" ").join(
            [
                sql.SQL("INSERT INTO"),
                sql.Identifier(property_object_md_schema.name),
                sql.SQL("("),
                sql.SQL(",").join(map(sql.Identifier, columns)),
                sql.SQL(") VALUES ("),
                sql.SQL(",").join(sql.Placeholder() * len(columns)),
                sql.SQL(")"),
                sql.SQL(
                    "ON CONFLICT (hash) DO UPDATE SET multiplicity = {}.multiplicity + 1;"
                ).format(sql.Identifier(property_object_md_schema.name)),
            ]
        )
        return sql_compose

    @staticmethod
    def co_row_to_values(row_dict):
        name = row_dict["names"][0]
        dataset_id = row_dict["dataset_ids"][0]
        vals = [row_dict.get(k) for k in config_md_schema.columns]
        vals.append(dataset_id)
        vals.append(name)
        return vals

    @staticmethod
    def po_row_to_values(row_dict):
        vals = [row_dict.get(k) for k in property_object_md_schema.columns]
        return vals

    def load_data_in_batches(
        self,
        configs,
        dataset_id=None,
        prop_map=None,
    ):
        """Load data to PostgreSQL in batches."""

        co_po_rows = self.gather_co_po_in_batches(configs, dataset_id, prop_map)
        co_sql = self.get_co_sql()
        po_sql = self.get_po_sql()
        for co_po_batch in tqdm(
            co_po_rows,
            desc="Loading data to database: ",
            unit="batch",
        ):
            co_rows, po_rows = list(zip(*co_po_batch))

            if len(co_rows) == 0:
                continue
            # TODO: Need to modify dataset.to_row_dict to properly aggregate values and get data to get two copies
            # TODO: Ensure all columns are present here
            # TODO: get column names from query and ensure len matches values
            # columns = list(zip(*self.get_table_schema("property_objects")))[0]
            co_values = map(self.co_row_to_values, co_rows)
            po_values = map(self.po_row_to_values, po_rows)
            with psycopg.connect(
                dbname=self.dbname,
                user=self.user,
                port=self.port,
                host=self.host,
                password=self.password,
            ) as conn:
                with conn.cursor() as curs:
                    curs.executemany(co_sql, co_values)
                    curs.executemany(po_sql, po_values)

    def create_table(self, schema):
        name_type = [
            (sql.Identifier(column.name), sql.SQL(column.type))
            for column in schema.columns
        ]
        query = sql.SQL(" ").join(
            [
                sql.SQL("CREATE TABLE IF NOT EXISTS"),
                sql.Identifier(schema.name),
                sql.SQL("("),
                sql.SQL(",").join(
                    [sql.SQL(" ").join([name, type_]) for name, type_ in name_type]
                ),
                sql.SQL(")"),
            ]
        )
        self.execute_sql(query)

    def create_ds_table(self):
        self.create_table(dataset_schema)

    # currently cf-kit table with some properties removed
    def create_po_table(self):
        self.create_table(property_object_md_schema)

    def create_co_table(self):
        self.create_table(config_md_schema)

    def create_pd_table(self):
        self.create_table(property_definition_schema)

    def insert_property_definition(self, property_dict):
        # TODO: try except that property_dict must be jsonable
        json_pd = json.dumps(property_dict)
        last_modified = get_last_modified()
        md5_hash = hashlib.md5(json_pd.encode()).hexdigest()
        sql = """
            INSERT INTO property_definitions (hash, last_modified, definition)
            VALUES (%s, %s, %s)
            ON CONFLICT (hash)
            DO NOTHING
        """
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            port=self.port,
            host=self.host,
            password=self.password,
        ) as conn:
            with conn.cursor() as curs:
                curs.execute(sql, (md5_hash, last_modified, json_pd))
        # TODO: insert columns into po table
        for key, v in property_dict.items():
            if key in [
                "property-id",
                "property-name",
                "property-title",
                "property-description",
            ]:
                continue
            else:
                column_name = property_dict["property-name"].replace(
                    "-", "_"
                ) + f"_{key}".replace("-", "_")
                if v["type"] == "float":
                    data_type = "DOUBLE PRECISION"
                elif v["type"] == "int":
                    data_type = "INT"
                elif v["type"] == "bool":
                    data_type = "BOOL"
                else:
                    data_type = "VARCHAR (10000)"
                for i in range(len(v["extent"])):
                    data_type += "[]"
            try:
                self.insert_new_column("property_objects", column_name, data_type)

            except Exception as e:
                print(f"An error occurred: {e}")

    def get_property_definitions(self):
        sql = """
             SELECT definition
             FROM property_definitions;
        """
        defs = self.general_query(sql)
        dict_defs = []
        for d in defs:
            dict_defs.append(json.loads(d["definition"]))
        return dict_defs

    def insert_data_and_create_dataset(
        self,
        configs,
        name: str,
        authors: list[str],
        description: str,
        publication_link: str = None,
        data_link: str = None,
        dataset_id: str = None,
        other_links: list[str] = None,
        publication_year: str = None,
        doi: str = None,
        labels: list[str] = None,
        data_license: str = "CC-BY-4.0",
        config_table=None,
        prop_object_table=None,
        prop_map=None,
    ):

        if dataset_id is None:
            dataset_id = generate_ds_id()

        converted_configs = []
        for c in configs:
            if isinstance(c, Atoms):
                converted_configs.append(AtomicConfiguration.from_ase(c))
            elif isinstance(c, AtomicConfiguration):
                converted_configs.append(c)
            else:
                raise Exception(
                    "Configs must be an instance of either ase.Atoms or AtomicConfiguration"  # noqa E501
                )

        self.load_data_in_batches(
            converted_configs, dataset_id, config_table, prop_object_table, prop_map
        )
        self.create_dataset(
            name,
            dataset_id,
            authors,
            publication_link,
            data_link,
            description,
            other_links,
            publication_year,
            doi,
            labels,
            data_license,
        )
        return dataset_id

    def get_table_schema(self, table_name):

        # Query to get the table schema
        query = """
        SELECT
            column_name,
            data_type,
            character_maximum_length,
            is_nullable
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
        """
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            port=self.port,
            host=self.host,
            password=self.password,
        ) as conn:
            with conn.cursor() as curs:
                curs.execute(query, (table_name,))
                schema = curs.fetchall()
                return schema

    def create_dataset(
        self,
        name: str,
        dataset_id: str,
        authors: list[str],
        publication_link: str,
        data_link: str,
        description: str,
        other_links: list[str] = None,
        publication_year: str = None,
        doi: str = None,
        labels: list[str] = None,
        data_license: str = "CC-BY-4.0",
    ):
        # find cs_ids, co_ids, and pi_ids
        config_df = self.dataset_query(dataset_id, "configurations")
        prop_df = self.dataset_query(dataset_id, "property_objects")

        if isinstance(authors, str):
            authors = [authors]
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
            doi=doi,
            data_license=data_license,
            configuration_set_ids=None,
            publication_year=publication_year,
        )
        row = ds.row_dict

        sql = """
            INSERT INTO datasets (last_modified, nconfigurations, nproperty_objects, nsites, nelements, elements, total_elements_ratio, nperiodic_dimensions, dimension_types, energy_mean, energy_variance, atomic_forces_count, cauchy_stress_count, energy_count, authors, description, license, links, name, publication_year, doi, id, extended_id, hash, labels)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s)
            ON CONFLICT (hash)
            DO NOTHING
        """

        column_headers = tuple(row.keys())
        values = []
        t = []
        for column in column_headers:
            if column in ["nconfiguration_sets"]:
                pass
            else:
                val = row[column]
                t.append(val)
            values.append(t)

        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            port=self.port,
            host=self.host,
            password=self.password,
        ) as conn:
            with conn.cursor() as curs:
                curs.executemany(sql, values)

    def insert_new_column(self, table, column_name, data_type):
        sql = f"""
            ALTER TABLE {table}
            ADD COLUMN {column_name} {data_type};
        """
        self.execute_sql(sql)

    def update_dataset(self, configs, dataset_id, prop_map):
        # convert to CF AtomicConfiguration if not already
        converted_configs = []
        for c in configs:
            if isinstance(c, Atoms):
                converted_configs.append(AtomicConfiguration.from_ase(c))
            elif isinstance(c, AtomicConfiguration):
                converted_configs.append(c)
            else:
                raise Exception(
                    "Configs must be an instance of either ase.Atoms or AtomicConfiguration"  # noqa E501
                )
        # update dataset_id
        # TODO: Change so it iterates from largest version
        v_no = dataset_id.split("_")[-1]
        new_v_no = int(v_no) + 1
        new_dataset_id = (
            dataset_id.split("_")[0]
            + "_"
            + dataset_id.split("_")[1]
            + "_"
            + str(new_v_no)
        )

        self.load_data_in_batches(converted_configs, new_dataset_id, prop_map=prop_map)

        # config_df_1 = self.dataset_query(dataset_id, 'configurations')
        # prop_df_1 = self.dataset_query(dataset_id, 'property_objects')

        config_df_2 = self.dataset_query(new_dataset_id, "configurations")
        prop_df_2 = self.dataset_query(new_dataset_id, "property_objects")

        # config_df_1.extend(config_df_2)
        # prop_df_1.extend(prop_df_2)

        old_ds = self.get_dataset(dataset_id)[0]

        # format links
        s = old_ds["links"][0].split(" ")[-1].replace("'", "")
        d = old_ds["links"][1].split(" ")[-1].replace("'", "")
        o = old_ds["links"][2].split(" ")[-1].replace("'", "")

        ds = Dataset(
            name=old_ds["name"],
            authors=old_ds["authors"],
            config_df=config_df_2,
            prop_df=prop_df_2,
            publication_link=s,
            data_link=d,
            description=old_ds["description"],
            other_links=o,
            dataset_id=new_dataset_id,
            labels=old_ds["labels"],
            doi=old_ds["doi"],
            data_license=old_ds["license"],
            # TODO handle cs later
            configuration_set_ids=None,
            publication_year=old_ds["publication_year"],
        )
        row = ds.row_dict

        sql = """
            INSERT INTO datasets (last_modified, nconfigurations, nproperty_objects, nsites, nelements, elements, total_elements_ratio, nperiodic_dimensions, dimension_types, energy_mean, energy_variance, atomic_forces_count, cauchy_stress_count, energy_count, authors, description, license, links, name, publication_year, doi, id, extended_id, hash, labels)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s)
            ON CONFLICT (hash)
            DO NOTHING
        """

        column_headers = tuple(row.keys())
        values = []
        t = []
        for column in column_headers:
            if column in ["nconfiguration_sets"]:
                pass
            else:
                val = row[column]
                t.append(val)
            values.append(t)

        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            port=self.port,
            host=self.host,
            password=self.password,
        ) as conn:
            with conn.cursor() as curs:
                curs.executemany(sql, values)
                return new_dataset_id

    def get_dataset_data(self, dataset_id):
        sql = f"""
        SELECT
            c.*,
            po.*
        FROM
            (SELECT * FROM configurations WHERE '{dataset_id}' = ANY(dataset_ids)) c
        INNER JOIN
            (SELECT * FROM property_objects WHERE dataset_id = '{dataset_id}') po
        ON
            c.id = po.configuration_id;
        """
        return self.general_query(sql)

    def general_query(self, sql):
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            port=self.port,
            host=self.host,
            password=self.password,
            row_factory=dict_row,
        ) as conn:
            with conn.cursor() as curs:
                curs.execute(sql)
                try:
                    return curs.fetchall()
                except:
                    return

    def execute_sql(self, sql):
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            port=self.port,
            host=self.host,
            password=self.password,
        ) as conn:
            with conn.cursor() as curs:
                curs.execute(sql)

    def dataset_query(
        self,
        dataset_id=None,
        table_name=None,
    ):
        if table_name == "configurations":
            sql = f"""
                SELECT *
                FROM {table_name}
                WHERE '{dataset_id}' = ANY(dataset_ids);
            """
        elif table_name == "property_objects":
            sql = f"""
                SELECT *
                FROM {table_name}
                WHERE dataset_id = '{dataset_id}';
            """
        else:
            raise Exception(
                "Only configurations and property_objects tables are supported"
            )

        return self.general_query(sql)

    def get_dataset(self, dataset_id):
        sql = f"""
                SELECT *
                FROM datasets
                WHERE id = '{dataset_id}';
            """
        print(dataset_id)
        return self.general_query(sql)

    def create_configuration_sets(
        self,
        loader,
        name_label_match: list[tuple],
    ):
        """
        Args for name_label_match in order:
        1. String pattern for matching CONFIGURATION NAMES
        2. String pattern for matching CONFIGURATION LABELS
        3. Name for configuration set
        4. Description for configuration set
        """
        dataset_id = self.dataset_id
        config_set_rows = []
        for i, (names_match, label_match, cs_name, cs_desc) in tqdm(
            enumerate(name_label_match), desc="Creating Configuration Sets"
        ):
            print(
                f"names match: {names_match}, label: {label_match}, "
                f"cs_name: {cs_name}, cs_desc: {cs_desc}"
            )
            config_set_query_df = loader.config_set_query(
                query_table=loader.config_table,
                dataset_id=dataset_id,
                name_match=names_match,
                label_match=label_match,
            )
            co_id_df = (
                config_set_query_df.select("id")
                .distinct()
                .withColumnRenamed("id", "configuration_id")
            )
            string_cols = [
                "elements",
            ]
            unstring_col_udf = sf.udf(unstring_df_val, ArrayType(StringType()))
            for col in string_cols:
                config_set_query_df = config_set_query_df.withColumn(
                    col, unstring_col_udf(sf.col(col))
                )
            unstring_col_udf = sf.udf(unstring_df_val, ArrayType(IntegerType()))
            int_cols = [
                "atomic_numbers",
                "dimension_types",
            ]
            for col in int_cols:
                config_set_query_df = config_set_query_df.withColumn(
                    col, unstring_col_udf(sf.col(col))
                )
            t = time()
            prelim_cs_id = f"CS_{cs_name}_{self.dataset_id}"
            co_cs_df = loader.get_co_cs_mapping(prelim_cs_id)
            if co_cs_df is not None:
                print(
                    f"Configuration Set {cs_name} already exists.\nRemove rows matching "  # noqa E501
                    f"'configuration_set_id == {prelim_cs_id} from table {loader.co_cs_map_table} to recreate.\n"  # noqa E501
                )
                continue
            config_set = ConfigurationSet(
                name=cs_name,
                description=cs_desc,
                config_df=config_set_query_df,
                dataset_id=self.dataset_id,
            )
            co_cs_df = co_id_df.withColumn("configuration_set_id", sf.lit(config_set.id))
            loader.write_table(co_cs_df, loader.co_cs_map_table, check_unique=False)
            loader.update_existing_co_rows(
                co_df=config_set_query_df,
                cols=["configuration_set_ids"],
                elems=config_set.id,
            )
            t_end = time() - t
            print(f"Time to create CS and update COs with CS-ID: {t_end}")

            config_set_rows.append(config_set.row_dict)
        config_set_df = loader.spark.createDataFrame(
            config_set_rows, schema=configuration_set_schema
        )
        loader.write_table(config_set_df, loader.config_set_table)
        return config_set_rows

    def delete_dataset(self, dataset_id):
        sql = """
            DELETE
            FROM datasets
            WHERE id = %s;
        """
        # TODO: delete children as well
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            port=self.port,
            host=self.host,
            password=self.password,
        ) as conn:
            with conn.cursor() as curs:
                curs.execute(sql, (dataset_id,))


class S3BatchManager:
    def __init__(self, bucket_name, access_id, secret_key, endpoint_url=None):
        self.bucket_name = bucket_name
        self.access_id = access_id
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url
        self.client = self.get_client()
        self.MAX_BATCH_SIZE = 100

    def get_client(self):
        return boto3.client(
            "s3",
            use_ssl=False,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_id,
            aws_secret_access_key=self.secret_key,
            region_name="fake-region",
            config=boto3.session.Config(
                signature_version="s3v4", s3={"addressing_style": "path"}
            ),
        )

    def batch_write(self, file_batch):
        results = []
        for key, content in file_batch:
            try:
                self.client.put_object(Bucket=self.bucket_name, Key=key, Body=content)
                results.append((key, None))
            except Exception as e:
                results.append((key, str(e)))
        return results


def write_md_partition(partition, config):
    s3_mgr = S3BatchManager(
        bucket_name=config["bucket_dir"],
        access_id=config["access_key"],
        secret_key=config["access_secret"],
        endpoint_url=config["endpoint"],
    )
    file_batch = []
    for row in partition:
        md_path = Path(config["metadata_dir"]) / row["metadata_path"]
        file_batch.append((str(md_path), row["metadata"]))

        if len(file_batch) >= s3_mgr.MAX_BATCH_SIZE:
            _ = s3_mgr.batch_write(file_batch)
            file_batch = []
    if file_batch:
        _ = s3_mgr.batch_write(file_batch)
    return iter([])


class S3FileManager:
    def __init__(self, bucket_name, access_id, secret_key, endpoint_url=None):
        self.bucket_name = bucket_name
        self.access_id = access_id
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url

    def get_client(self):
        return boto3.client(
            "s3",
            use_ssl=False,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_id,
            aws_secret_access_key=self.secret_key,
            region_name="fake-region",
            config=boto3.session.Config(
                signature_version="s3v4", s3={"addressing_style": "path"}
            ),
        )

    def write_file(self, content, file_key):
        try:
            client = self.get_client()
            client.put_object(Bucket=self.bucket_name, Key=file_key, Body=content)
            # return (f"/vdev/{self.bucket_name}/{file_key}", sys.getsizeof(content))
        except Exception as e:
            return f"Error: {str(e)}"

    def read_file(self, file_key):
        try:
            client = self.get_client()
            # key = file_key.replace(str(Path("/vdev/colabfit-data")) + "/", "")
            response = client.get_object(Bucket=self.bucket_name, Key=file_key)
            return response["Body"].read().decode("utf-8")
        except Exception as e:
            return f"Error: {str(e)}"


def generate_ds_id():
    # Maybe check to see whether the DS ID already exists?
    ds_id = ID_FORMAT_STRING.format("DS", generate_string(), 0)
    # print("Generated new DS ID:", ds_id)
    return ds_id


"""
@sf.udf(returnType=StringType())
def prepend_path_udf(prefix, md_path):
    try:
        full_path = Path(prefix) / Path(md_path).relative_to("/")
        return str(full_path)
    except ValueError:
        full_path = Path(prefix) / md_path
        return str(full_path)
"""

# def write_md_partition(partition, config):
#     s3_mgr = S3FileManager(
#         bucket_name=config["bucket_dir"],
#         access_id=config["access_key"],
#         secret_key=config["access_secret"],
#         endpoint_url=config["endpoint"],
#     )
#     for row in partition:
#         md_path = Path(config["metadata_dir"]) / row["metadata_path"]
#         if not md_path.exists():
#             s3_mgr.write_file(
#                 row["metadata"],
#                 str(md_path),
#             )
#     return iter([])


def read_md_partition(partition, config):
    s3_mgr = S3FileManager(
        bucket_name=config["bucket_dir"],
        access_id=config["access_key"],
        secret_key=config["access_secret"],
        endpoint_url=config["endpoint"],
    )

    def process_row(row):
        rowdict = row.asDict()
        try:
            rowdict["metadata"] = s3_mgr.read_file(row["metadata_path"])
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                rowdict["metadata"] = None
            else:
                print(f"Error reading {row['metadata_path']}: {str(e)}")
                rowdict["metadata"] = None
        return Row(**rowdict)

    return map(process_row, partition)


'''
def dataset_query(
    dataset_id=None,
    table_name=None,
):
    if table_name == 'configurations':
        sql = f"""
            SELECT *
            FROM {table_name}
            WHERE '{dataset_id}' = ANY(dataset_ids);
        """
    elif table_name == 'property_objects':
        sql = f"""
            SELECT *
            FROM {table_name}
            WHERE dataset_id = '{dataset_id}';
        """
    else:
        raise Exception('Only configurations and property_objects tables are supported')

    with psycopg.connect(dbname=self.dbname, user=self.user, port=self.port, host=self.host, password=self.password,row_factory=dict_row) as conn:
        with conn.cursor() as curs:
            r = curs.execute(sql)
            return curs.fetchall()

def get_dataset(dataset_id):
    sql = f"""
            SELECT *
            FROM datasets
            WHERE id = '{dataset_id}';
        """

    with psycopg.connect(dbname=self.dbname, user=self.user, port=self.port, host=self.host, password=self.password,row_factory=dict_row) as conn:
        with conn.cursor() as curs:
            r = curs.execute(sql)
            return curs.fetchall()
'''
