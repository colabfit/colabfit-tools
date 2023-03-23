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
        do_hash = sha512()
        do_hash.update(self.configuration.encode('utf-8'))
        for i in sorted(self.properties):
            do_hash.update(str(i).encode('utf-8'))

        return int(do_hash.hexdigest(), 16)

    def __str__(self):
        return "DataObject(Configuration='{}', Number of Attached PropertyInstances={})".format(
            self.configuration,
            len(self.properties),
        )

    def __repr__(self):
        return str(self)
