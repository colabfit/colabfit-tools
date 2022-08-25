from hashlib import sha512
from collections import OrderedDict

class Metadata:
    """
        A Metadata object contains key-value pairs of generic information which can be
        attached to Configurations, ?ConfigurationSets and DataSets?

        Attributes:

            metadata (dict):
                A dict of all metadata

    """

    def __init__(self,metadata):
        self.metadata = metadata
        self._hash = hash(self)

    def __hash__(self):
        """
        Hashes Metadata information
        """
        _hash = sha512()
        od = OrderedDict(sorted(self.metadata.items()))
        for k, v in od.items():
            _hash.update(k.encode('utf-8'))
            if isinstance(v,(list,set,tuple)):
                for i in v:
                    if isinstance(i,str):
                        _hash.update(i.encode('utf-8'))
                    else:
                        _hash.update(bytes(i))
            elif isinstance(v,str):
                _hash.update(v.encode('utf-8'))
            else:
                _hash.update(bytes(v))
        return int(_hash.hexdigest(), 16)