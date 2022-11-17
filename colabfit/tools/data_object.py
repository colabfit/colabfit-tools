from hashlib import sha512



class DataObject:
    """
    A DataObject groups a Configuration with one or more PropertyInstances.

    Attributes:

        configuration (str):
            The configuration's ID

        properties (list):
             A list of all attached property instance IDs
    """

    def __init__(self, configuration, properties):
        self.configuration = configuration
        self.properties = properties
        self._hash = hash(self)

    def __hash__(self):
        hash = sha512()
        hash.update(self.configuration.encode('utf-8'))
        for i in sorted(self.properties):
            hash.update(str(i).encode('utf-8'))

        return int(hash.hexdigest(), 16)

    def __str__(self):
        return "DataObject(Configuration='{}', Number of Attached PropertyInstances={})".format(
            self.configuration,
            len(self.properties),
        )

    def __repr__(self):
        return str(self)
