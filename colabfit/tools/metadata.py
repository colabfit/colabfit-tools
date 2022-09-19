from hashlib import sha512
from collections import OrderedDict

class Metadata:
    """
        A Metadata object contains key-value pairs of generic information which can be
        attached to Configurations, PropertyInstances, ConfigurationSets and DataSets.

        Attributes:
            linked_type (str)
                Identity of linked object: must be one of ["CO","PI","CS","DS"]
            metadata (dict):
                A dict of arbitrary metadata

    """

    def __init__(self, linked_type=None, metadata=None):
        if linked_type not in ["CO", "PI", "CS", "DS"]:
            raise RuntimeError('linked_type must be one of CO, PI, CS, or DS')
        self.linked_type = linked_type
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
            if isinstance(v['source-value'],(list,set,tuple)):
                for i in v['source-value']:
                    if isinstance(i,str):
                        _hash.update(i.encode('utf-8'))
                    else:
                        _hash.update(bytes(i))
            elif isinstance(v['source-value'],str):
                _hash.update(v['source-value'].encode('utf-8'))
            else:
                _hash.update(bytes(v['source-value']))
            if 'source-units' in v:
                _hash.update(v['source-unit'].encode('utf-8'))
        return int(_hash.hexdigest(), 16)

    def __str__(self):
        return str(self.metadata)
