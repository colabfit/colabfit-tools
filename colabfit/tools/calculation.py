from hashlib import sha512



class Calculation:
    """
    A Calculation object groups a Configuration with one or more PropertyInstances.

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
        cs_hash = sha512()
        cs_hash.update(self.configuration.encode('utf-8'))
        for i in sorted(self.properties):
            cs_hash.update(str(i).encode('utf-8'))

        return int(cs_hash.hexdigest(), 16)

    def __str__(self):
        return "Calculation(Configuration='{}', Number of Attached PropertyInstances={})".format(
            self.configuration,
            len(self.properties),
        )

    def __repr__(self):
        return str(self)
