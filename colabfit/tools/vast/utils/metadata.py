import json
import os
import sys

import numpy as np

from .hashing import _hash


def _sort_dict(dictionary: dict) -> dict:
    keys = list(dictionary.keys())
    keys.sort()
    return {k: dictionary[k] for k in keys}


def _parse_unstructured_metadata(md_json: dict) -> dict:
    if md_json == {}:
        return {
            "metadata": None,
            "metadata_id": None,
            "metadata_path": None,
            "metadata_size": None,
        }
    md = {}
    for key, val in md_json.items():
        if key in ["_id", "hash", "colabfit-id", "last_modified", "software", "method"]:
            continue
        if isinstance(val, dict):
            if "source-value" in val.keys():
                val = val["source-value"]
        if type(val).__module__ == np.__name__:
            val = getattr(val, "tolist", lambda: val)()
        if isinstance(val, list) and len(val) == 1:
            val = val[0]
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, dict):
            val = _sort_dict(val)
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        md[key] = val
    md = _sort_dict(md)
    md_hash = str(_hash(md, md.keys(), include_keys_in_hash=True))
    md["hash"] = md_hash
    md["id"] = f"MD_{md_hash[:25]}"
    split = md["id"][-4:]
    filename = f"{md['id']}.json"
    after_bucket = os.path.join(split, filename)
    metadata = json.dumps(md)
    return {
        "metadata": metadata,
        "metadata_id": md["id"],
        "metadata_path": after_bucket,
        "metadata_size": sys.getsizeof(metadata),
    }
