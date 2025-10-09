from logging import getLogger


def get_session():
    from dotenv import load_dotenv
    from vastdb.session import Session
    import os

    load_dotenv()
    access = os.getenv("VAST_DB_ACCESS")
    secret = os.getenv("VAST_DB_SECRET")
    endpoint = os.getenv("VAST_DB_ENDPOINT")
    return Session(access=access, secret=secret, endpoint=endpoint)


def append_wip_table_to_prod(from_table, to_table, dry_run=False):
    session = get_session()
    logger = getLogger(__name__)
    total_rows = 0
    _, from_bucket, from_schema, from_tbl = from_table.split(".")
    _, to_bucket, to_schema, to_tbl = to_table.split(".")
    with session.transaction() as tx:
        from_table = tx.bucket(from_bucket).schema(from_schema).table(from_tbl)
        to_table = tx.bucket(to_bucket).schema(to_schema).table(to_tbl)
        rdr = from_table.select()
        logger.info(f"Reading from {from_table}")
        logger.info(f"Writing to {to_table}")
        if dry_run:
            for batch in rdr:
                logger.info(batch.num_rows)
                total_rows += batch.num_rows
        else:
            for batch in rdr:
                logger.info(batch.num_rows)
                total_rows += batch.num_rows
                to_table.insert(batch)
    logger.info(f"total_rows: {total_rows}")
