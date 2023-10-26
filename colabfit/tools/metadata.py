from hashlib import sha512
from collections import OrderedDict
import numpy as np


class Metadata:
    """
    A Metadata object contains key-value pairs of generic information which can be
    attached to Configurations, PropertyInstances, ConfigurationSets and DataSets.

    Attributes:
        metadata (dict):
            A dict of arbitrary metadata

    """

    def __init__(self, metadata=None):
        self.metadata = metadata
        self._hash = hash(self)

    @classmethod
    def from_map(cls, d, source):
        gathered_fields = {}
        for md_field in d.keys():
            if "value" in d[md_field]:
                v = d[md_field]["value"]
            elif "field" in d[md_field]:
                field_key = d[md_field]["field"]

                if field_key in source.info:
                    v = source.info[field_key]
                elif field_key in source.arrays:
                    v = source.arrays[field_key]
                else:
                    # No keys are required; ignored if missing
                    continue
            else:
                # No keys are required; ignored if missing
                continue

            if "units" in d[md_field]:
                gathered_fields[md_field] = {
                    "source-value": v,
                    "source-unit": d[md_field]["units"],
                }
            else:
                gathered_fields[md_field] = {"source-value": v}
        return cls(metadata=gathered_fields)

    def __hash__(self):
        """
        Hashes Metadata information
        """
        _hash = sha512()
        od = OrderedDict(sorted(self.metadata.items()))
        for k, v in od.items():
            _hash.update(k.encode("utf-8"))
            if isinstance(v["source-value"], (list, set, tuple)):
                for i in v["source-value"]:
                    if isinstance(i, str):
                        _hash.update(i.encode("utf-8"))
                    else:
                        _hash.update(np.array(i).data.tobytes())
            elif isinstance(v["source-value"], str):
                _hash.update(v["source-value"].encode("utf-8"))
            else:
                _hash.update(np.array(v["source-value"]).data.tobytes())
            if "source-units" in v:
                _hash.update(v["source-unit"].encode("utf-8"))
        return int(_hash.hexdigest(), 16)

    def __str__(self):
        return str(self.metadata)
