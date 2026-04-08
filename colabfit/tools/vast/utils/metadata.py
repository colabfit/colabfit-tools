import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

METADATA_MAX_CHARS = 10_000


def _sort_dict(dictionary: dict) -> dict:
    keys = list(dictionary.keys())
    keys.sort()
    return {k: dictionary[k] for k in keys}


def _parse_unstructured_metadata(md_json: dict) -> dict:
    if md_json == {}:
        return {"metadata": None}
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
    md_str = json.dumps(md)
    if len(md_str) > METADATA_MAX_CHARS:
        raise ValueError(
            f"Metadata exceeds maximum allowed size of {METADATA_MAX_CHARS} characters "
            f"({len(md_str)} chars). Reduce metadata content before ingesting."
        )
    return {"metadata": md_str}
